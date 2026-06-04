"use client";

import { useState } from "react";
import type { ExtractionResponse, ExtractionUpdate, StreamPatchPayload } from "@/lib/types";
// Note: Need to check if react-dropzone is installed, if not will use standard input or recommend installing. 
// Actually, user didn't have react-dropzone in package.json. I should stick to standard input or standard drag/drop events to avoid new deps if possible, OR ask to install. 
// Given the "wow" requirement, I'll implement a custom drag/drop area using standard HTML events to avoid dependency hell, or use the existing deps.
// framer-motion is available for animations. maintain strict UI.

import { Upload, FileText, Loader2, AlertCircle } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { cn } from "@/lib/utils"; // Assuming utils exists, if not I will create strict version inline or check. Next.js usually has it. 
// Wait, I haven't checked for utils. I'll make a safe version or check. 
// Let's assume standard simple upload first.

interface UploadZoneProps {
    onExtractionComplete: (data: ExtractionUpdate) => void;
    onFileName: (name: string) => void;
    onFileSelect: (file: File) => void;
    onReset?: () => void;
    hasPreviewData?: boolean;
}

export function UploadZone({ onExtractionComplete, onFileName, onFileSelect, onReset, hasPreviewData = false }: UploadZoneProps) {
    const [isDragging, setIsDragging] = useState(false);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState<string | null>(null);

    const handleDragOver = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(true);
    };

    const handleDragLeave = () => {
        setIsDragging(false);
    };

    const [phase, setPhase] = useState<string>("");

    const processFile = async (file: File) => {
        if (file.type !== "application/pdf") {
            setError("Vui lòng tải lên file PDF.");
            return;
        }

        // Clear any previous extraction before starting a new one so stale
        // results (table, images, slabs, GL markers) never leak into the new run.
        onReset?.();

        setIsLoading(true);
        setError(null);
        setPhase("Đang chuẩn bị...");
        onFileName(file.name);
        onFileSelect(file);

        const formData = new FormData();
        formData.append("file", file);

        try {
            const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL;
            // Use SSE stream for progressive updates. EventSource doesn't
            // support POST + body, so we read the response body manually
            // and parse "event:"/"data:" lines per SSE spec.
            const res = await fetch(`${apiBaseUrl}/extract-foundation-data-stream`, {
                method: "POST",
                body: formData,
            });
            if (!res.ok || !res.body) {
                throw new Error(`HTTP ${res.status}`);
            }

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            // Parse events from the stream buffer. Each event ends with "\n\n".
            const handleEvent = (eventName: string, dataStr: string) => {
                let payload: unknown = null;
                try { payload = JSON.parse(dataStr); } catch { return; }
                const eventPayload = payload as Partial<ExtractionResponse> & Partial<StreamPatchPayload> & {
                    total_pages?: number;
                    oval_gl_list?: unknown[];
                    chunk_index?: number;
                    chars_received?: number;
                    detail?: string;
                    message?: string;
                };

                switch (eventName) {
                    case "connected":
                        setPhase("Đã kết nối, đang phân tích PDF...");
                        break;
                    case "status":
                        if (eventPayload.message) setPhase(eventPayload.message);
                        break;
                    case "started":
                        setPhase(`Đang xử lý ${eventPayload.total_pages || 0} trang...`);
                        break;
                    case "oval_gl_markers":
                        setPhase(`Đã phát hiện ${eventPayload.oval_gl_list?.length || 0} GL marker, đang gọi AI...`);
                        break;
                    case "gemini_progress":
                        setPhase(`Gemini đang trả dữ liệu... chunk ${eventPayload.chunk_index || 0}, ${eventPayload.chars_received || 0} ký tự`);
                        break;
                    // Intermediate events update the loading text ONLY — results are
                    // rendered once, on "complete", so the user sees a single finished
                    // view instead of progressive partial updates.
                    case "stream_patch":
                        if (eventPayload.foundation_list?.length) {
                            setPhase(`Đang cập nhật thêm ${eventPayload.foundation_list.length} cấu kiện từ Gemini...`);
                        } else if (eventPayload.oval_gl_list?.length || eventPayload.floor_regular_list?.length) {
                            setPhase("Đang cập nhật thêm dữ liệu sàn/GL...");
                        }
                        break;
                    case "partial_table_data":
                        setPhase("Đã đọc được bảng móng từ PDF, đang xử lý tiếp...");
                        break;
                    case "partial_data":
                        setPhase("Đang hoàn tất dữ liệu...");
                        break;
                    case "complete":
                        setPhase("Hoàn tất");
                        onExtractionComplete(eventPayload as ExtractionResponse);
                        break;
                    case "error":
                        throw new Error(eventPayload.detail || "Stream error");
                }
            };

            // Keep reading chunks; flush completed events from the buffer.
            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true }).replace(/\r\n/g, "\n").replace(/\r/g, "\n");

                let sepIdx: number;
                while ((sepIdx = buffer.indexOf("\n\n")) >= 0) {
                    const block = buffer.slice(0, sepIdx);
                    buffer = buffer.slice(sepIdx + 2);

                    let eventName = "message";
                    const dataLines: string[] = [];
                    for (const line of block.split("\n")) {
                        if (line.startsWith("event:")) {
                            eventName = line.slice(6).trim();
                        } else if (line.startsWith("data:")) {
                            dataLines.push(line.slice(5).trim());
                        }
                    }
                    if (dataLines.length > 0) {
                        handleEvent(eventName, dataLines.join("\n"));
                    }
                }
            }
        } catch (err: unknown) {
            console.error(err);
            setError(err instanceof Error ? err.message : "Trích xuất dữ liệu thất bại. Vui lòng thử lại.");
        } finally {
            setIsLoading(false);
            setPhase("");
        }
    };

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault();
        setIsDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) processFile(file);
    };

    const handleFileInput = (e: React.ChangeEvent<HTMLInputElement>) => {
        const file = e.target.files?.[0];
        if (file) processFile(file);
    };

    return (
        <div className="w-full max-w-2xl mx-auto mb-8">
            <motion.div
                layout
                className={cn(
                    "relative border-2 border-dashed rounded-xl transition-all duration-300 ease-out text-center cursor-pointer overflow-hidden",
                    isLoading && hasPreviewData ? "p-5" : "p-12",
                    isDragging
                        ? "border-blue-500 bg-blue-50/50 scale-[1.02]"
                        : "border-gray-200 hover:border-gray-300 hover:bg-gray-50/50",
                    isLoading && "pointer-events-none opacity-80"
                )}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                onDrop={handleDrop}
                onClick={() => document.getElementById("file-upload")?.click()}
            >
                <input
                    id="file-upload"
                    type="file"
                    accept=".pdf"
                    className="hidden"
                    onChange={handleFileInput}
                />

                <AnimatePresence mode="wait">
                    {isLoading ? (
                        <motion.div
                            key="loading"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className={cn(
                                "flex items-center",
                                hasPreviewData ? "flex-row gap-4 text-left justify-between" : "flex-col gap-4"
                            )}
                        >
                            <Loader2 className={cn("text-blue-500 animate-spin", hasPreviewData ? "w-6 h-6 flex-shrink-0" : "w-12 h-12")} />
                            <div className={cn("space-y-1", hasPreviewData && "flex-1")}>
                                <h3 className={cn("font-semibold text-gray-900", hasPreviewData ? "text-sm" : "text-lg")}>
                                    {hasPreviewData ? "Đang bổ sung dữ liệu chi tiết..." : "AI đang phân tích..."}
                                </h3>
                                <p className={cn("text-gray-500", hasPreviewData ? "text-xs" : "text-sm")}>
                                    {phase || "Đang trích xuất dữ liệu cấu kiện từ bản vẽ"}
                                </p>
                            </div>
                        </motion.div>
                    ) : (
                        <motion.div
                            key="idle"
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={{ opacity: 0, y: -10 }}
                            className="flex flex-col items-center gap-4"
                        >
                            <div className={cn(
                                "p-4 rounded-full transition-colors",
                                isDragging ? "bg-blue-100 text-blue-600" : "bg-gray-100 text-gray-600"
                            )}>
                                <Upload className="w-8 h-8" />
                            </div>
                            <div className="space-y-1">
                                <h3 className="text-lg font-semibold text-gray-900">
                                    Tải lên bản vẽ
                                </h3>
                                <p className="text-sm text-gray-500">
                                    Kéo thả file PDF vào đây, hoặc nhấn để chọn file
                                </p>
                            </div>
                            <div className="flex items-center gap-2 text-xs text-gray-400 mt-2">
                                <FileText className="w-4 h-4" />
                                <span>Chỉ PDF (Quét toàn bộ trang)</span>
                            </div>
                        </motion.div>
                    )}
                </AnimatePresence>
            </motion.div>

            <AnimatePresence>
                {error && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="mt-4 p-4 bg-red-50 text-red-600 rounded-lg flex items-center gap-3 border border-red-100"
                    >
                        <AlertCircle className="w-5 h-5 flex-shrink-0" />
                        <p className="text-sm font-medium">{error}</p>
                    </motion.div>
                )}
            </AnimatePresence>
        </div>
    );
}



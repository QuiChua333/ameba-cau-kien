"use client";

import { ExtractionResponse, PitHoleItem } from "@/lib/types";
import { motion, AnimatePresence } from "framer-motion";
import { useState, useEffect } from "react";
import { X, ZoomIn, ZoomOut, Maximize2, Download } from "lucide-react";
import { cn } from "@/lib/utils";

// --- Image Modal Component ---
function ImageModal({ src, alt, onClose }: { src: string; alt: string; onClose: () => void }) {
    const [scale, setScale] = useState(1);

    const handleZoomIn = (e: React.MouseEvent) => {
        e.stopPropagation();
        setScale(prev => Math.min(prev + 0.5, 5));
    };

    const handleZoomOut = (e: React.MouseEvent) => {
        e.stopPropagation();
        setScale(prev => Math.max(prev - 0.5, 0.5));
    };

    useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            if (e.key === "Escape") {
                onClose();
            }
        };
        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [onClose]);

    return (
        <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/90 backdrop-blur-sm"
            onClick={onClose}
        >
            {/* Controls */}
            <div className="absolute top-4 right-4 flex items-center gap-2 z-50">
                <button
                    onClick={handleZoomOut}
                    className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-full transition-colors"
                    title="Thu nhỏ"
                >
                    <ZoomOut className="w-6 h-6" />
                </button>
                <button
                    onClick={handleZoomIn}
                    className="p-2 bg-white/10 hover:bg-white/20 text-white rounded-full transition-colors"
                    title="Phóng to"
                >
                    <ZoomIn className="w-6 h-6" />
                </button>
                <button
                    onClick={onClose}
                    className="p-2 bg-white/10 hover:bg-red-500/20 text-white hover:text-red-400 rounded-full transition-colors ml-2"
                    title="Đóng"
                >
                    <X className="w-8 h-8" />
                </button>
            </div>

            {/* Image Container with Scroll for high zoom */}
            <div className="w-full h-full overflow-auto flex items-center justify-center p-8" onClick={(e) => e.stopPropagation()}>
                 <motion.img
                    initial={{ scale: 0.8, opacity: 0 }}
                    animate={{ scale: scale, opacity: 1 }}
                    src={src}
                    alt={alt}
                    className="max-w-[90vw] max-h-[90vh] object-contain transition-transform duration-200 ease-out cursor-move"
                    style={{ transformOrigin: "center center" }}
                    drag
                    dragConstraints={{ left: -500, right: 500, top: -500, bottom: 500 }}
                />
            </div>
            
            <div className="absolute bottom-6 left-1/2 -translate-x-1/2 bg-black/50 px-4 py-2 rounded-full text-white text-sm pointer-events-none">
                Cuộn/Kéo để di chuyển • Độ phóng: {Math.round(scale * 100)}%
            </div>
        </motion.div>
    );
}

function Badge({ children, className }: { children: React.ReactNode; className?: string }) {
    return (
        <span className={cn("px-2.5 py-0.5 rounded-full text-xs font-semibold border", className)}>
            {children}
        </span>
    );
}

function TopElevationBadge({ value, alt }: { value?: number | null; alt?: number | null }) {
    if (value === null || value === undefined) {
        return <span className="text-gray-400 text-xs italic">N/A</span>;
    }
    const label = value === 0 ? "▽GL ±0" : value > 0 ? `▽GL +${value}` : `▽GL ${value}`;
    const className = value < 0
        ? "bg-red-50 text-red-600 border-red-200"
        : value > 0
        ? "bg-green-50 text-green-700 border-green-200"
        : "bg-gray-100 text-gray-600 border-gray-200";
    const altLabel = alt === 0 ? "▽GL ±0" : (alt ?? 0) > 0 ? `▽GL +${alt}` : `▽GL ${alt}`;
    return (
        <div className="flex flex-col gap-1">
            <Badge className={className}>{label} mm</Badge>
            {alt !== null && alt !== undefined && (
                <span
                    className="text-[10px] font-medium text-red-600"
                    title="Mặt cắt (断面) khác mặt bằng (伏図) — đang lấy giá trị mặt bằng, cần kiểm tra"
                >
                    ⚠ Mặt cắt: {altLabel} mm
                </span>
            )}
        </div>
    );
}

function ClassificationBadge({ type }: { type: string }) {
    switch (type) {
        case "DD":
            return <Badge className="bg-purple-100 text-purple-700 border-purple-200">DD</Badge>;
        case "D":
            return <Badge className="bg-blue-100 text-blue-700 border-blue-200">D</Badge>;
        case "TNF":
            return <Badge className="bg-gray-100 text-gray-700 border-gray-200">TNF</Badge>;
        case "FW/FG":
            return <Badge className="bg-orange-100 text-orange-700 border-orange-200">FW/FG</Badge>;
        default:
            return <Badge className="bg-gray-100 text-gray-700 border-gray-200">{type}</Badge>;
    }
}

interface ResultsTableProps {
    data: ExtractionResponse;
    pdfFile?: File | null;
}

export function ResultsTable({ data, pdfFile }: ResultsTableProps) {
    const { project_info, foundation_list } = data;
    const [selectedImage, setSelectedImage] = useState<{ src: string; alt: string } | null>(null);
    const [pdfUrl, setPdfUrl] = useState<string | null>(null);
    const [excelLoading, setExcelLoading] = useState(false);
    const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL;

    const downloadExcel = async () => {
        if (!apiBaseUrl) {
            alert("Thiếu cấu hình NEXT_PUBLIC_API_URL.");
            return;
        }

        setExcelLoading(true);
        try {
            const res = await fetch(`${apiBaseUrl}/generate-excel`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    foundation_list,
                    pit_list: data.pit_list ?? [],
                }),
            });

            if (!res.ok) {
                let errorMessage = `HTTP ${res.status}`;
                const contentType = res.headers.get("content-type") || "";

                if (contentType.includes("application/json")) {
                    const errorBody = await res.json();
                    errorMessage = errorBody.detail || errorBody.message || errorMessage;
                } else {
                    const errorText = await res.text();
                    errorMessage = errorText || errorMessage;
                }

                throw new Error(errorMessage);
            }

            const blob = await res.blob();
            const url = URL.createObjectURL(blob);
            const contentDisposition = res.headers.get("content-disposition") || "";
            const utf8FileNameMatch = contentDisposition.match(/filename\*=UTF-8''([^;]+)/i);
            const asciiFileNameMatch = contentDisposition.match(/filename=\"?([^\";]+)\"?/i);
            const fileName = utf8FileNameMatch?.[1]
                ? decodeURIComponent(utf8FileNameMatch[1])
                : asciiFileNameMatch?.[1] || "計算書(施工) - DD.xlsx";
            const a = document.createElement("a");
            a.href = url;
            a.download = fileName;
            document.body.appendChild(a);
            a.click();
            a.remove();
            URL.revokeObjectURL(url);
        } catch (err) {
            const message = err instanceof Error ? err.message : String(err);
            alert("Excel download failed: " + message);
        } finally {
            setExcelLoading(false);
        }
    };

    // Create Object URL for PDF
    useEffect(() => {
        if (pdfFile) {
            const url = URL.createObjectURL(pdfFile);
            setPdfUrl(url);
            return () => URL.revokeObjectURL(url);
        } else {
            setPdfUrl(null);
        }
    }, [pdfFile]);

    // Separate foundations and beams
    const foundations = foundation_list.filter(item => item.classification !== "FW/FG");
    const beams = foundation_list.filter(item => item.classification === "FW/FG");
    // GL markers — listed flat (no regular/sloped floor distinction).
    const glMarkers = data.oval_gl_list || [];
    const glMarkerGroups = Object.entries(
        glMarkers.reduce<Record<string, number>>((acc, m) => {
            acc[m.text] = (acc[m.text] || 0) + 1;
            return acc;
        }, {})
    ).map(([text, count]) => ({ text, count }));
    const hasFloorData = !!(data.floor_overview_base64 || glMarkers.length);
    const canDownloadExcel = !!data.excel_ready && foundation_list.length > 0;
    const pitList: PitHoleItem[] = data.pit_list ?? [];

    return (
        <>
            <AnimatePresence>
                {selectedImage && (
                    <ImageModal 
                        src={selectedImage.src} 
                        alt={selectedImage.alt} 
                        onClose={() => setSelectedImage(null)} 
                    />
                )}
            </AnimatePresence>

            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-full space-y-8"
            >
            {data.is_partial && (
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                    <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-amber-900">
                        <p className="text-sm font-semibold">Đang hiển thị dữ liệu tạm</p>
                        <p className="text-sm text-amber-800">
                            {data.partial_message || "Gemini đang bổ sung các trường còn thiếu. Bảng sẽ tự cập nhật khi xong."}
                        </p>
                    </div>
                </div>
            )}

            {/* Project Info Card */}
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm flex flex-wrap gap-8 items-center bg-gradient-to-r from-gray-50 to-white">
                    <div>
                        <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Tên dự án</h3>
                        <p className="text-lg font-bold text-gray-900">{project_info.project_name || "Chưa có thông tin"}</p>
                    </div>
                    <div>
                        <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Tỉ lệ</h3>
                        <p className="text-sm font-semibold text-gray-900 bg-gray-100 px-3 py-1 rounded-md border border-gray-200">
                            {project_info.drawing_scale || "N/A"}
                        </p>
                    </div>
                    <div>
                        <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-1">Ngày</h3>
                        <p className="text-sm font-medium text-gray-700">{project_info.drawing_date || "N/A"}</p>
                    </div>
                </div>
            </div>

            {/* ========== FOUNDATION SECTION ========== */}
            {foundations.length > 0 && (
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
                    <div className="flex items-center gap-3 flex-wrap">
                        <div className="h-8 w-1 bg-blue-500 rounded-full"></div>
                        <h2 className="text-2xl font-bold text-gray-900">Móng (基礎)</h2>
                        <span className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">{foundations.length} mục</span>
                        {canDownloadExcel && (
                            <button
                                onClick={downloadExcel}
                                disabled={excelLoading}
                                className="ml-auto flex items-center gap-2 px-4 py-2 bg-emerald-600 hover:bg-emerald-700 disabled:opacity-60 text-white text-sm font-semibold rounded-lg shadow-sm transition-all"
                            >
                                <Download className="w-4 h-4" />
                                {excelLoading ? "Đang tạo..." : "Tải Excel"}
                            </button>
                        )}
                    </div>

                    {data.excel_ready && data.images_pending && (
                        <div className="rounded-xl border border-emerald-200 bg-emerald-50 px-4 py-3 text-emerald-900">
                            <p className="text-sm font-semibold">Đã đủ dữ liệu số để tải Excel</p>
                            <p className="text-sm text-emerald-800">
                                Bạn có thể tải Excel ngay bây giờ. Ảnh crop và ảnh tổng quan vẫn đang được xử lý ở nền.
                            </p>
                        </div>
                    )}

                    {/* Foundation Table Image */}
                    {data.table_image_base64 && (
                        <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
                            <h3 className="text-xs font-medium text-gray-500 uppercase tracking-wider mb-4">Bảng gốc từ bản vẽ</h3>
                            <div 
                                className="overflow-hidden rounded-lg border border-gray-100 cursor-pointer hover:ring-2 hover:ring-blue-400 transition-all"
                                onClick={() => setSelectedImage({ 
                                    src: `data:image/jpeg;base64,${data.table_image_base64}`, 
                                    alt: "Foundation Table" 
                                })}
                            >
                                <img 
                                    src={`data:image/jpeg;base64,${data.table_image_base64}`} 
                                    alt="Extracted Table Region" 
                                    className="w-full h-auto object-contain max-h-[500px]"
                                />
                            </div>
                        </div>
                    )}

                    {/* Foundation Data Table (No Images) */}
                    <div className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden">
                        <div className="overflow-x-auto">
                            <table className="w-full text-sm text-left">
                                <thead className="text-xs text-gray-500 uppercase bg-gray-50 border-b border-gray-200">
                                    <tr>
                                        <th scope="col" className="px-6 py-4 font-medium min-w-[100px]">Ký hiệu</th>
                                        <th scope="col" className="px-4 py-4 font-medium text-center min-w-[110px]">Ảnh</th>
                                        <th scope="col" className="px-6 py-4 font-medium text-center">Phân loại</th>
                                        <th scope="col" className="px-6 py-4 font-medium">Kích thước (mm)</th>
                                        <th scope="col" className="px-6 py-4 font-medium">Cao độ đỉnh (▽GL)</th>
                                        <th scope="col" className="px-6 py-4 font-medium">Cốt thép (X)</th>
                                        <th scope="col" className="px-6 py-4 font-medium">Cốt thép (Y)</th>
                                        <th scope="col" className="px-6 py-4 font-medium min-w-[200px]">Ghi chú</th>
                                    </tr>
                                </thead>
                                <tbody className="divide-y divide-gray-100">
                                    {foundations.map((item, index) => (
                                        <motion.tr
                                            key={index}
                                            initial={{ opacity: 0, x: -10 }}
                                            animate={{ opacity: 1, x: 0 }}
                                            transition={{ delay: index * 0.05 }}
                                            className="hover:bg-gray-50/80 transition-colors"
                                        >
                                            <td className="px-6 py-4 font-bold text-gray-900">{item.type}</td>
                                            <td className="px-4 py-3">
                                                {item.image_base64 ? (
                                                    <button
                                                        type="button"
                                                        onClick={() => setSelectedImage({
                                                            src: `data:image/jpeg;base64,${item.image_base64}`,
                                                            alt: item.type,
                                                        })}
                                                        className="relative group block w-20 h-20 rounded-md border border-gray-200 bg-gray-50 overflow-hidden hover:ring-2 hover:ring-blue-400 transition-all"
                                                        title={`Phóng to ảnh ${item.type}`}
                                                    >
                                                        <img
                                                            src={`data:image/jpeg;base64,${item.image_base64}`}
                                                            alt={item.type}
                                                            className="w-full h-full object-contain p-1"
                                                        />
                                                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                                                            <Maximize2 className="w-5 h-5 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow" />
                                                        </div>
                                                    </button>
                                                ) : (
                                                    <div className="w-20 h-20 rounded-md border border-dashed border-gray-200 bg-gray-50 flex items-center justify-center">
                                                        <span className="text-[10px] text-gray-400 text-center px-1 leading-tight">Không có ảnh</span>
                                                    </div>
                                                )}
                                            </td>
                                            <td className="px-6 py-4 text-center">
                                                <ClassificationBadge type={item.classification} />
                                            </td>
                                            <td className="px-6 py-4">
                                                <div className="flex flex-col gap-1 text-gray-600">
                                                    <span className="flex items-center gap-2">
                                                        <span className="w-6 text-xs text-gray-400">Lx</span>
                                                        <span className="font-mono font-medium text-gray-900">{item.dimensions.Lx}</span>
                                                    </span>
                                                    <span className="flex items-center gap-2">
                                                        <span className="w-6 text-xs text-gray-400">Ly</span>
                                                        <span className="font-mono font-medium text-gray-900">{item.dimensions.Ly}</span>
                                                    </span>
                                                    <span className="flex items-center gap-2">
                                                        <span className="w-6 text-xs text-gray-400">D</span>
                                                        <span className="font-mono text-gray-900">{item.dimensions.D}</span>
                                                    </span>
                                                </div>
                                            </td>
                                            <td className={`px-6 py-4 ${item.top_elevation_alt != null ? "bg-red-50" : ""}`}>
                                                <TopElevationBadge value={item.top_elevation} alt={item.top_elevation_alt} />
                                            </td>
                                            <td className="px-6 py-4 font-mono text-gray-700">{item.rebar_x}</td>
                                            <td className="px-6 py-4 font-mono text-gray-700">{item.rebar_y}</td>
                                            <td className="px-6 py-4 text-gray-500 italic max-w-xs truncate" title={item.remarks}>
                                                {item.remarks || "-"}
                                            </td>
                                        </motion.tr>
                                    ))}
                                </tbody>
                            </table>
                        </div>
                    </div>
                </div>
            )}

            {/* ========== BEAM SECTION ========== */}
            {beams.length > 0 && (
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
                    <div className="flex items-center gap-3">
                        <div className="h-8 w-1 bg-orange-500 rounded-full"></div>
                        <h2 className="text-2xl font-bold text-gray-900">Dầm móng (梁)</h2>
                        <span className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">{beams.length} mục</span>
                    </div>

                    {/* Beam Cards Grid */}
                    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                        {beams.map((beam, index) => (
                            <motion.div
                                key={index}
                                initial={{ opacity: 0, scale: 0.95 }}
                                animate={{ opacity: 1, scale: 1 }}
                                transition={{ delay: index * 0.1 }}
                                className="bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden hover:shadow-md transition-shadow"
                            >
                                {/* Beam Image */}
                                {beam.image_base64 ? (
                                    <div 
                                        className="relative group cursor-pointer bg-gray-50"
                                        onClick={() => setSelectedImage({ 
                                            src: `data:image/jpeg;base64,${beam.image_base64}`, 
                                            alt: beam.type 
                                        })}
                                    >
                                        <img 
                                            src={`data:image/jpeg;base64,${beam.image_base64}`} 
                                            alt={beam.type}
                                            className="w-full h-48 object-contain p-4 transition-transform group-hover:scale-105"
                                        />
                                        <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                                            <Maximize2 className="w-8 h-8 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg" />
                                        </div>
                                    </div>
                                ) : (
                                    <div className="w-full h-48 bg-gray-100 flex items-center justify-center">
                                        <span className="text-gray-400 text-sm">Không có ảnh</span>
                                    </div>
                                )}

                                {/* Beam Info */}
                                <div className="p-4 space-y-3">
                                    <div className="flex items-center justify-between">
                                        <h3 className="text-lg font-bold text-gray-900">{beam.type}</h3>
                                        <ClassificationBadge type={beam.classification} />
                                    </div>

                                    {/* Top Elevation + Beam Height row */}
                                    <div className="flex flex-wrap items-center gap-3">
                                        <div className="flex items-center gap-2">
                                            <span className="text-xs text-gray-400 font-medium uppercase tracking-wide">Cao độ đỉnh</span>
                                            <TopElevationBadge value={beam.top_elevation} />
                                        </div>
                                        {beam.dimensions.D && Number(beam.dimensions.D) > 0 && (
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs text-gray-400 font-medium uppercase tracking-wide">D</span>
                                                <Badge className="bg-orange-50 text-orange-700 border-orange-200">
                                                    {beam.dimensions.D} mm
                                                </Badge>
                                            </div>
                                        )}
                                    </div>

                                    {/* Remarks if available */}
                                    {beam.remarks && beam.remarks !== "-" && (
                                        <p className="text-xs text-gray-500 italic border-t border-gray-100 pt-2">
                                            {beam.remarks}
                                        </p>
                                    )}
                                </div>
                            </motion.div>
                        ))}
                    </div>
                </div>
            )}

        {/* ========== PIT HOLE SECTION ========== */}
        {pitList.length > 0 && (
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
                <div className="flex items-center gap-3">
                    <div className="h-8 w-1 bg-violet-500 rounded-full"></div>
                    <h2 className="text-2xl font-bold text-gray-900">Hố Pit (ピット)</h2>
                    <span className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                        {pitList.length} hố
                    </span>
                    <span className="text-xs text-emerald-700 bg-emerald-50 border border-emerald-200 px-2 py-1 rounded-full">
                        {pitList.filter(p => p.readable).length} đọc được → Excel
                    </span>
                    {pitList.filter(p => !p.readable).length > 0 && (
                        <span className="text-xs text-amber-700 bg-amber-50 border border-amber-200 px-2 py-1 rounded-full">
                            {pitList.filter(p => !p.readable).length} không đọc được
                        </span>
                    )}
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                    {pitList.map((pit, index) => (
                        <motion.div
                            key={index}
                            initial={{ opacity: 0, scale: 0.95 }}
                            animate={{ opacity: 1, scale: 1 }}
                            transition={{ delay: index * 0.08 }}
                            className={`bg-white rounded-xl border shadow-sm overflow-hidden hover:shadow-md transition-shadow ${
                                pit.readable ? "border-gray-200" : "border-amber-200"
                            }`}
                        >
                            {/* Pit Image */}
                            {pit.image_base64 ? (
                                <div
                                    className="relative group cursor-pointer bg-gray-50"
                                    onClick={() => setSelectedImage({
                                        src: `data:image/jpeg;base64,${pit.image_base64}`,
                                        alt: pit.type,
                                    })}
                                >
                                    <img
                                        src={`data:image/jpeg;base64,${pit.image_base64}`}
                                        alt={pit.type}
                                        className="w-full h-48 object-contain p-4 transition-transform group-hover:scale-105"
                                    />
                                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/10 transition-colors flex items-center justify-center">
                                        <Maximize2 className="w-8 h-8 text-white opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-lg" />
                                    </div>
                                </div>
                            ) : (
                                <div className="w-full h-48 bg-gray-100 flex items-center justify-center">
                                    <span className="text-gray-400 text-sm">Không có ảnh</span>
                                </div>
                            )}

                            {/* Pit Info */}
                            <div className="p-4 space-y-3">
                                <div className="flex items-center justify-between gap-2">
                                    <h3 className="text-base font-bold text-gray-900 truncate">{pit.type}</h3>
                                    {pit.readable ? (
                                        <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200 shrink-0">Đọc được</Badge>
                                    ) : (
                                        <Badge className="bg-amber-50 text-amber-700 border-amber-200 shrink-0">Không đọc được</Badge>
                                    )}
                                </div>

                                {pit.readable ? (
                                    <div className="flex flex-wrap items-center gap-3">
                                        <div className="flex items-center gap-2">
                                            <span className="text-xs text-gray-400 font-medium uppercase tracking-wide">Cao độ đỉnh</span>
                                            <TopElevationBadge value={pit.top_elevation} />
                                        </div>
                                        {pit.D != null && pit.D > 0 && (
                                            <div className="flex items-center gap-2">
                                                <span className="text-xs text-gray-400 font-medium uppercase tracking-wide">D</span>
                                                <Badge className="bg-violet-50 text-violet-700 border-violet-200">
                                                    {pit.D} mm
                                                </Badge>
                                            </div>
                                        )}
                                    </div>
                                ) : (
                                    <p className="text-xs text-amber-600 italic">
                                        Không xác định được thông số — chỉ hiển thị, không ghi vào Excel.
                                    </p>
                                )}
                            </div>
                        </motion.div>
                    ))}
                </div>
            </div>
        )}

        {/* ========== FLOOR DETECTION SECTION ========== */}
        {hasFloorData && (
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 space-y-6">
                <div className="flex items-center gap-3">
                    <div className="h-8 w-1 bg-teal-500 rounded-full"></div>
                    <h2 className="text-2xl font-bold text-gray-900">Ký hiệu GL</h2>
                    <span className="text-sm text-gray-500 bg-gray-100 px-3 py-1 rounded-full">
                        {glMarkers.length} marker
                    </span>
                </div>

                <div className="grid grid-cols-1 lg:grid-cols-[1.3fr_0.9fr] gap-6">
                    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm space-y-4">
                        <div className="grid grid-cols-2 gap-3">
                            <StatCard label="GL Marker" value={glMarkers.length} />
                            <StatCard label="Slab" value={data.slab_list?.length || 0} />
                        </div>

                        {glMarkerGroups.length > 0 && (
                            <div className="space-y-3">
                                <h3 className="text-sm font-semibold text-gray-900">Danh sách marker</h3>
                                <div className="flex flex-wrap gap-2">
                                    {glMarkerGroups.map((m) => (
                                        <Badge key={m.text} className="bg-teal-50 text-teal-700 border-teal-200">
                                            {m.text}{m.count > 1 ? ` • ${m.count}` : ""}
                                        </Badge>
                                    ))}
                                </div>
                            </div>
                        )}

                        {!data.floor_overview_base64 && data.is_partial && (
                            <p className="text-sm text-gray-500">
                                Dữ liệu GL đã về trước. Ảnh tổng quan sẽ xuất hiện sau khi backend xử lý crop xong.
                            </p>
                        )}
                    </div>

                    <div className="bg-white rounded-xl border border-gray-200 p-6 shadow-sm">
                        {data.floor_overview_base64 ? (
                            <>
                                <div
                                    className="relative group cursor-pointer overflow-hidden rounded-lg bg-gray-50 border border-gray-100"
                                    onClick={() => setSelectedImage({
                                        src: `data:image/jpeg;base64,${data.floor_overview_base64}`,
                                        alt: "Floor Overview with GL Markers"
                                    })}
                                >
                                    <img
                                        src={`data:image/jpeg;base64,${data.floor_overview_base64}`}
                                        alt="Floor Overview"
                                        className="w-full h-auto object-contain max-h-[800px] transition-transform duration-500 group-hover:scale-[1.01]"
                                    />
                                    <div className="absolute inset-0 bg-black/0 group-hover:bg-black/5 transition-colors flex items-center justify-center pointer-events-none">
                                        <Maximize2 className="w-12 h-12 text-white/80 opacity-0 group-hover:opacity-100 transition-opacity drop-shadow-md" />
                                    </div>
                                </div>
                                <p className="text-sm text-gray-500 mt-3 text-center">
                                    Ký hiệu GL được đánh dấu đỏ. Nhấn để phóng to.
                                </p>
                            </>
                        ) : (
                            <div className="h-full min-h-60 rounded-lg border border-dashed border-gray-200 bg-gray-50 flex items-center justify-center text-center text-sm text-gray-400 px-6">
                                Chưa có ảnh tổng quan sàn. Backend đang hoàn tất bước annotate/crop.
                            </div>
                        )}
                    </div>
                </div>
            </div>
        )}

            {/* ========== PDF VIEWER SECTION ========== */}
            {/* REMOVED: Page Visualization of GL Markers as per user request */}
            
            {/* ========== FULL PDF VIEWER ========== */}
             <div className="space-y-6 pt-10 border-t border-gray-200 px-4">
                <div className="max-w-7xl mx-auto flex items-center gap-3">
                    <div className="h-8 w-1 bg-gray-800 rounded-full"></div>
                    <h2 className="text-2xl font-bold text-gray-900">Xem PDF đầy đủ</h2>
                </div>
                
                {/* Wider Container for PDF with slight margin and taller height */}
                <div className="w-full max-w-[85%] mx-auto bg-white rounded-xl border border-gray-200 shadow-sm overflow-hidden h-[900px]">
                    {pdfUrl ? (
                        <iframe 
                            src={pdfUrl} 
                            className="w-full h-full" 
                            title="PDF Viewer"
                        />
                    ) : (
                         <div className="w-full h-full flex items-center justify-center bg-gray-50 text-gray-400">
                            Không có file PDF để xem trước
                        </div>
                    )}
                </div>
            </div>

        </motion.div>
        </>
    );
}

function StatCard({ label, value }: { label: string; value: number }) {
    return (
        <div className="bg-white rounded-lg p-4 border border-gray-200 shadow-sm flex flex-col items-center justify-center">
            <span className="text-2xl font-bold text-gray-900">{value}</span>
            <span className="text-xs font-medium text-gray-500 uppercase">{label}</span>
        </div>
    );
}



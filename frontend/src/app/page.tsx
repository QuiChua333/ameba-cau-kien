"use client";

import { useEffect, useRef, useState } from "react";
import { UploadZone } from "@/components/UploadZone";
import { ResultsTable } from "@/components/ResultsTable";
import {
    ExtractionResponse,
    ExtractionUpdate,
    FoundationItem,
    OvalGLItem,
    ProjectInfo,
    RegularFloor,
    SlabItem,
    SlopedFloor,
    ItemRegion,
    StreamPatchPayload,
} from "@/lib/types";

function regionKey(region?: ItemRegion) {
    if (!region) return "no-region";
    return `${region.page}:${region.xmin}:${region.ymin}:${region.xmax}:${region.ymax}`;
}

function upsertByKey<T>(current: T[], incoming: T[], getKey: (item: T) => string): T[] {
    const map = new Map(current.map((item) => [getKey(item), item]));
    for (const item of incoming) {
        map.set(getKey(item), item);
    }
    return Array.from(map.values());
}

function sortFoundationItems(items: FoundationItem[]): FoundationItem[] {
    return [...items].sort((a, b) => {
        const parse = (value: string) => {
            const first = value.split(/[,、，\s]+/)[0].trim().toUpperCase();
            const normalized = first.replace(/[！-～]/g, (char) =>
                String.fromCharCode(char.charCodeAt(0) - 0xfee0)
            );
            const cat = normalized.startsWith("FG") ? 2 : normalized.startsWith("FW") ? 1 : 0;
            const match = normalized.match(/^F(?:[WG])?(\d+)([A-Z]*)/);
            return {
                cat,
                num: match ? Number(match[1]) : 9999,
                suffix: match?.[2] || normalized,
            };
        };
        const pa = parse(a.type);
        const pb = parse(b.type);
        return pa.cat - pb.cat || pa.num - pb.num || pa.suffix.localeCompare(pb.suffix);
    });
}

function mergeProjectInfo(current: ProjectInfo, incoming?: ProjectInfo): ProjectInfo {
    if (!incoming) return current;
    return {
        project_name: incoming.project_name && incoming.project_name !== "Đang trích xuất..." ? incoming.project_name : current.project_name,
        drawing_date: incoming.drawing_date && incoming.drawing_date !== "N/A" ? incoming.drawing_date : current.drawing_date,
        drawing_scale: incoming.drawing_scale && incoming.drawing_scale !== "N/A" ? incoming.drawing_scale : current.drawing_scale,
    };
}

function isPatch(update: ExtractionUpdate): update is StreamPatchPayload {
    return "is_patch" in update && update.is_patch === true;
}

function buildExtractionFromPatch(update: StreamPatchPayload): ExtractionResponse {
    return {
        project_info: update.project_info || {
            project_name: "Đang trích xuất...",
            drawing_date: "N/A",
            drawing_scale: "N/A",
        },
        foundation_list: sortFoundationItems(update.foundation_list || []),
        slab_list: update.slab_list || [],
        oval_gl_list: update.oval_gl_list || [],
        annotated_pages: [],
        floor_regular_list: update.floor_regular_list || [],
        floor_sloped_list: update.floor_sloped_list || [],
        is_partial: true,
        partial_stage: "gemini_stream",
        partial_message: update.partial_message,
        excel_ready: update.excel_ready ?? false,
        images_pending: update.images_pending ?? true,
    };
}

export default function Home() {
    const [data, setData] = useState<ExtractionResponse | null>(null);
    const [fileName, setFileName] = useState<string | null>(null);
    const [pdfFile, setPdfFile] = useState<File | null>(null);
    const resultsRef = useRef<HTMLDivElement | null>(null);
    const hasAutoScrolledRef = useRef(false);

    useEffect(() => {
        if (!data) {
            hasAutoScrolledRef.current = false;
            return;
        }
        if (!resultsRef.current || hasAutoScrolledRef.current) return;
        resultsRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
        hasAutoScrolledRef.current = true;
    }, [data]);

    const handleReset = () => {
        setData(null);
        setFileName(null);
        setPdfFile(null);
        hasAutoScrolledRef.current = false;
    };

    const handleExtractionUpdate = (update: ExtractionUpdate) => {
        setData((current) => {
            if (!current) {
                return isPatch(update) ? buildExtractionFromPatch(update) : update;
            }

            if (!isPatch(update)) {
                return {
                    ...update,
                    project_info: mergeProjectInfo(current.project_info, update.project_info),
                    foundation_list: sortFoundationItems(update.foundation_list),
                    table_image_base64: update.table_image_base64 || current.table_image_base64,
                    slab_overview_base64: update.slab_overview_base64 || current.slab_overview_base64,
                    floor_overview_base64: update.floor_overview_base64 || current.floor_overview_base64,
                    partial_message: update.partial_message || current.partial_message,
                    excel_ready: update.excel_ready ?? current.excel_ready,
                    images_pending: update.images_pending ?? current.images_pending,
                };
            }

            const nextFoundations = update.foundation_list
                ? sortFoundationItems(upsertByKey<FoundationItem>(current.foundation_list, update.foundation_list, (item) => item.type))
                : current.foundation_list;
            const nextSlabs = update.slab_list
                ? upsertByKey<SlabItem>(current.slab_list, update.slab_list, (item) => `${item.type}:${regionKey(item.region)}`)
                : current.slab_list;
            const nextOvals = update.oval_gl_list
                ? upsertByKey<OvalGLItem>(current.oval_gl_list, update.oval_gl_list, (item) => `${item.text}:${regionKey(item.region)}`)
                : current.oval_gl_list;
            const nextRegularFloors = update.floor_regular_list
                ? upsertByKey<RegularFloor>(current.floor_regular_list, update.floor_regular_list, (item) => item.elevation)
                : current.floor_regular_list;
            const nextSlopedFloors = update.floor_sloped_list
                ? upsertByKey<SlopedFloor>(
                    current.floor_sloped_list,
                    update.floor_sloped_list,
                    (item) => `${item.start_elevation}:${item.end_elevation}:${regionKey(item.region)}`
                )
                : current.floor_sloped_list;

            return {
                ...current,
                project_info: mergeProjectInfo(current.project_info, update.project_info),
                foundation_list: nextFoundations,
                slab_list: nextSlabs,
                oval_gl_list: nextOvals,
                floor_regular_list: nextRegularFloors,
                floor_sloped_list: nextSlopedFloors,
                is_partial: true,
                partial_stage: "gemini_stream",
                partial_message: update.partial_message || current.partial_message,
                excel_ready: update.excel_ready ?? current.excel_ready,
                images_pending: update.images_pending ?? current.images_pending,
            };
        });
    };

    return (
        <div className="min-h-screen bg-white">
            {/* Header */}
            <header className="border-b border-gray-100 bg-white/80 backdrop-blur-md">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-16 flex items-center justify-between">
                    <div className="flex items-center gap-2">
                        <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center text-white font-bold">
                            F
                        </div>
                        <h1 className="text-xl font-bold text-gray-900 tracking-tight">Foundation<span className="text-blue-600">X</span></h1>
                    </div>
                </div>
            </header>

            <main className="w-full py-12">
                <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 mb-12 space-y-4 text-center">
                    <h2 className="text-4xl font-extrabold text-gray-900 tracking-tight sm:text-5xl">
                       Trích xuất thông tin cấu kiện
                    </h2>
                    <p className="max-w-2xl mx-auto text-lg text-gray-500">
                        Tải lên bản vẽ PDF để tự động trích xuất, phân loại và cấu trúc hóa dữ liệu cấu kiện.
                    </p>
                    {fileName && (
                        <div className="mt-4 animate-in fade-in slide-in-from-bottom-2">
                            <span className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-blue-50 text-blue-700 text-sm font-medium border border-blue-100">
                                📄 {fileName}
                            </span>
                        </div>
                    )}
                </div>

                <div className="flex flex-col items-center space-y-12">
                    <div className="w-full max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
                        <UploadZone
                            onExtractionComplete={handleExtractionUpdate}
                            onFileName={setFileName}
                            onFileSelect={setPdfFile}
                            onReset={handleReset}
                            hasPreviewData={!!data}
                        />
                    </div>
                    
                    {data && (
                        <div ref={resultsRef} className="w-full">
                            <ResultsTable data={data} pdfFile={pdfFile} />
                        </div>
                    )}
                    
                    {!data && (
                       <div className="text-sm text-gray-400 mt-20">
                            Powered by Claude AI & Vercel
                       </div>
                    )}
                </div>
            </main>
        </div>
    );
}

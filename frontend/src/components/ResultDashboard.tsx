"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { Download, ZoomIn, X } from "lucide-react"
import { cn } from "@/lib/utils"

interface ItemData {
  name: string
  type: string
}

interface APIResponse {
  status: string
  data: string[][]
  excel_url: string
  evidence_image: string | null
}

interface ResultDashboardProps {
  data: APIResponse
}

export default function ResultDashboard({ data }: ResultDashboardProps) {
  const [zoomedImage, setZoomedImage] = useState<boolean>(false)
  const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL

  const items: string[][] = data.data || []
  const foundations: ItemData[] = items.map((item) => ({
    name: item[0],
    type: item[1]
  }))

  return (
    <div className="w-full max-w-6xl mx-auto space-y-8 animate-in fade-in slide-in-from-bottom-5 duration-700">
      
      {/* Header Actions */}
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">Kết quả trích xuất</h2>
        <a 
          href={`${apiBaseUrl}${data.excel_url}`}
          className="flex items-center gap-2 px-6 py-2.5 rounded-lg bg-green-600 hover:bg-green-500 text-white font-medium shadow-lg shadow-green-500/20 transition-all"
        >
          <Download className="w-4 h-4" />
          Tải Excel
        </a>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
        
        {/* Left Column: Data Table */}
        <div className="glass-panel rounded-2xl p-6 h-fit max-h-[600px] overflow-y-auto custom-scrollbar">
          <div className="flex items-center justify-between mb-6 border-b border-white/10 pb-4">
             <h3 className="text-lg font-semibold text-indigo-300">Móng ({foundations.length})</h3>
          </div>

          {foundations.length > 0 ? (
             <div className="grid grid-cols-1 gap-3">
               {foundations.map((item, idx) => (
                 <div key={idx} className="flex justify-between items-center p-3 rounded-lg bg-white/5 border border-white/5 hover:bg-white/10 transition-colors">
                    <span className="font-mono text-white text-lg">{item.name}</span>
                    <span className={cn(
                      "px-3 py-1 rounded text-sm font-bold min-w-[60px] text-center",
                      item.type === 'DD' ? "bg-purple-500/20 text-purple-300 border border-purple-500/30" :
                      item.type === 'D' ? "bg-blue-500/20 text-blue-300 border border-blue-500/30" :
                      "bg-gray-500/20 text-gray-400 border border-gray-500/30"
                    )}>
                      {item.type}
                    </span>
                 </div>
               ))}
             </div>
          ) : (
             <div className="text-center py-10 text-gray-400">
               Không tìm thấy móng nào.
             </div>
          )}
        </div>

        {/* Right Column: Evidence Image */}
        <div className="space-y-4">
           <div className="glass-panel rounded-2xl p-4">
              <div className="flex items-center gap-2 mb-4">
                 <ZoomIn className="w-5 h-5 text-green-400" />
                 <h3 className="text-lg font-semibold text-white">Hình ảnh minh chứng</h3>
              </div>
              
              {data.evidence_image ? (
                <div className="relative group cursor-pointer" onClick={() => setZoomedImage(true)}>
                   <img 
                      src={`${apiBaseUrl}${data.evidence_image}`}
                      alt="Foundation Table"
                      className="w-full h-auto rounded-lg border border-white/10 hover:border-green-400 transition-all"
                   />
                   <div className="absolute inset-0 bg-black/0 group-hover:bg-black/20 rounded-lg transition-all flex items-center justify-center opacity-0 group-hover:opacity-100">
                      <ZoomIn className="w-12 h-12 text-white" />
                   </div>
                </div>
              ) : (
                <div className="h-64 flex items-center justify-center border-2 border-dashed border-white/10 rounded-xl text-gray-500">
                   Không có ảnh minh chứng.
                </div>
              )}
              
              <p className="text-xs text-gray-400 mt-2 text-center">
                 Nhấn để phóng to • Bảng móng được trích xuất từ PDF
              </p>
           </div>
        </div>

      </div>
      
      {/* Zoom Modal */}
      <AnimatePresence>
        {zoomedImage && data.evidence_image && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center p-4"
            onClick={() => setZoomedImage(false)}
          >
            <button 
              className="absolute top-4 right-4 p-2 rounded-full bg-white/10 hover:bg-white/20 text-white"
              onClick={() => setZoomedImage(false)}
            >
              <X className="w-6 h-6" />
            </button>
            <motion.img
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.8 }}
              src={`${apiBaseUrl}${data.evidence_image}`}
              alt="Zoomed Foundation Table"
              className="max-w-full max-h-full object-contain rounded-lg shadow-2xl"
              onClick={(e) => e.stopPropagation()}
            />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}

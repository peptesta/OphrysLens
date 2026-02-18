"use client";

import { useRef, ChangeEvent } from "react";
import { FolderUp, Loader2, ArrowRight } from "lucide-react";

// 1. Estendiamo l'interfaccia di React per supportare gli attributi delle cartelle
declare module 'react' {
  interface InputHTMLAttributes<T> extends HTMLAttributes<T> {
    webkitdirectory?: string;
    directory?: string;
  }
}

interface UploadSectionProps {
  fileCount: number;
  isTransforming: boolean;
  processedCount: number;
  onFilesSelected: (e: ChangeEvent<HTMLInputElement>) => void;
  onTransform: () => void;
}

export default function UploadSection({ 
  fileCount, 
  isTransforming, 
  processedCount, 
  onFilesSelected, 
  onTransform 
}: UploadSectionProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const percentage = fileCount > 0 
    ? Math.round((processedCount / fileCount) * 100) 
    : 0;

  return (
    <div className="animate-fade-in space-y-8">
      {/* UPLOAD CARD */}
      <div className="bg-white p-10 rounded-2xl shadow-lg shadow-stone-200/50 border border-[#D8D2C8] text-center">
        <div className="mb-6 flex justify-center">
          <div className="h-20 w-20 bg-[#F0F7F3] rounded-full flex items-center justify-center text-emerald-600">
            <FolderUp size={40} />
          </div>
        </div>
        <h2 className="text-xl font-bold text-stone-700 mb-2">Select Image Folder</h2>
        <p className="text-stone-500 mb-8 max-w-md mx-auto">
          Images will be sent to the server in batches.
        </p>
        <p className="text-stone-500 font-medium mt-1">
          You can use this tool in order to crop and customize the bounding box made by the AI cropper
        </p>
        
        <label className="relative inline-block group">
          <div className="bg-emerald-50 text-emerald-800 font-bold py-3 px-8 rounded-xl border-2 border-dashed border-emerald-200 group-hover:bg-emerald-100 group-hover:border-emerald-400 cursor-pointer transition flex items-center gap-2">
            Browse Folders...
          </div>
          {/* 2. Ora possiamo usare webkitdirectory e directory senza errori o 'any' */}
          <input 
            type="file" 
            ref={fileInputRef}
            webkitdirectory="" 
            directory=""
            multiple 
            onChange={onFilesSelected}
            className="absolute inset-0 w-full h-full opacity-0 cursor-pointer" 
          />
        </label>

        {fileCount > 0 && (
          <div className="mt-6 ml-4 p-4 bg-[#F0F7F3] rounded-xl inline-block border border-emerald-100">
            <p className="text-emerald-800 font-semibold">
              {fileCount} images loaded
            </p>
          </div>
        )}
      </div>

      {/* ACTION AREA / PROGRESS BAR */}
      <div className="flex justify-center w-full">
        {isTransforming ? (
          <div className="w-full md:w-[600px] bg-white p-6 rounded-xl shadow-lg border border-emerald-100 animate-in fade-in slide-in-from-bottom-2">
            <div className="flex justify-between items-center mb-2 gap-4">
              <span className="text-sm font-bold text-emerald-800 flex items-center gap-2 flex-shrink-0">
                <Loader2 size={16} className="animate-spin" /> 
                Processing...
              </span>

              <div className="flex items-center gap-3 text-sm font-bold text-emerald-600 font-mono whitespace-nowrap">
                <span>
                  {processedCount} / {fileCount} ({percentage}%)
                </span>
                <span className="text-stone-300">|</span>
                <span>
                  Tempo stimato: {Math.floor((fileCount * 2.5) / 60)}m {Math.round((fileCount * 2.5) % 60)}s      
                </span>
              </div>
            </div>
    
            <div className="w-full bg-stone-100 rounded-full h-4 overflow-hidden shadow-inner relative">
              <div 
                className="bg-emerald-500 h-full rounded-full flex items-center justify-center relative overflow-hidden"
                style={{ 
                  width: `${percentage}%`,
                  transition: "width 0.5s ease-out"
                }}
              >
                <div className="absolute inset-0 bg-white/20 animate-pulse" />
              </div>
            </div>
            
            <p className="text-xs text-stone-400 mt-2 text-center italic">
              Sending images to server... please wait.
            </p>
          </div>
        ) : (
          <button
            onClick={onTransform}
            disabled={fileCount === 0}
            className="w-full md:w-auto min-w-[200px] bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-4 px-8 rounded-xl shadow-lg shadow-emerald-200/50 transition-all transform active:scale-[0.98] disabled:opacity-50 disabled:cursor-not-allowed disabled:bg-stone-400 flex items-center justify-center gap-2 text-lg"
          >
            Transform <ArrowRight size={20} />
          </button>
        )}
      </div>
    </div>
  );
}
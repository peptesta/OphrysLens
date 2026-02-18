"use client";

import Image from "next/image";
import { Eye, ChevronLeft, ChevronRight, PenLine, RotateCcw } from "lucide-react";
import { ImageFile } from "../types"; 

interface ResultsListProps {
  files: ImageFile[];      
  totalCount: number;      
  currentPage: number;
  totalPages: number;
  startIndex: number;      
  endIndex: number;
  itemsPerPage: number;
  onSelect: (file: ImageFile) => void;
  onPageChange: (direction: 'next' | 'prev') => void;
  onResetAll: () => void; 
}

export default function ResultsList({
  files,
  totalCount,
  currentPage,
  totalPages,
  startIndex,
  endIndex,
  itemsPerPage,
  onSelect,
  onPageChange,
  onResetAll
}: ResultsListProps) {
  
  return (
    <div className="animate-fade-in space-y-6">
      <div className="bg-white rounded-2xl shadow-md shadow-stone-200/50 border border-stone-200 overflow-hidden flex flex-col min-h-[500px]">
        
        {/* List Header */}
        <div className="p-4 bg-[#F0F7F3] border-b border-[#D8D2C8] flex justify-between items-center flex-shrink-0">
          <h3 className="font-bold text-emerald-900 uppercase text-sm tracking-wider">
            Transformed Images ({totalCount})
          </h3>
          <span className="text-xs text-stone-500 italic">Click to view image and modify bounding box</span>
        </div>
        
        {/* Items List */}
        <div className="flex-1 overflow-y-auto">
          <ul className="divide-y divide-stone-100">
            {files.map((file, idx) => {
              const analysis = file.analysis;
              const totalBoxes = analysis?.count || 0;
              
              let changesCount = 0;
              if (totalBoxes > 0) {
                for (let i = 0; i < totalBoxes; i++) {
                   const isModified = analysis?.modified?.[i];
                   const isEliminated = analysis?.eliminated?.[i]; 

                   if (isModified || isEliminated) {
                      changesCount++;
                   }
                }
              }

              return (
                <li 
                  key={idx}
                  onClick={() => onSelect(file)}
                  className="group hover:bg-emerald-50/50 transition cursor-pointer p-4 flex items-center justify-between h-24" 
                >
                  <div className="flex items-center gap-6 truncate">
                    <div className="relative w-16 h-16 rounded-xl overflow-hidden border border-stone-200 bg-stone-100 flex-shrink-0 shadow-sm">
                      <Image src={file.url} alt="thumbnail" fill className="object-cover" unoptimized />
                    </div>
                    
                    <div className="truncate flex flex-col justify-center gap-1">
                      <p className="font-bold text-lg text-stone-700 group-hover:text-emerald-900 truncate">
                        {file.name}
                      </p>
                      
                      {totalBoxes > 0 && (
                        <p className="text-xs text-emerald-600 font-medium bg-emerald-50 px-2 py-0.5 rounded-md inline-block w-fit">
                          {totalBoxes} Detections Found
                        </p>
                      )}
                    </div>
                  </div>

                  <div className="flex items-center gap-6 pl-4 flex-shrink-0">
                    {changesCount > 0 && (
                      <span className="flex items-center gap-2 text-sm font-bold text-amber-700 bg-amber-100 px-3 py-1.5 rounded-lg border border-amber-200 animate-in fade-in shadow-sm">
                        <PenLine size={16} /> 
                        {changesCount} / {totalBoxes} Modified
                      </span>
                    )}

                    <div className="text-stone-300 group-hover:text-emerald-500 transition hover:scale-110">
                      <Eye size={28} />
                    </div>
                  </div>
                </li>
              );
            })}
            
            {files.length < itemsPerPage && (
                <div className="h-full bg-stone-50/30 flex-1" />
            )}
          </ul>
        </div>

        {/* Pagination & Global Actions */}
        <div className="p-4 border-t border-stone-200 bg-[#F0F7F3] flex items-center justify-between flex-shrink-0">
          
          {/* NEW: RESET ALL BUTTON */}
          <button 
            onClick={(e) => { e.stopPropagation(); onResetAll(); }}
            className="flex items-center gap-2 text-stone-500 hover:text-red-600 hover:bg-red-50 px-3 py-1.5 rounded-lg transition text-xs font-bold border border-transparent hover:border-red-200"
            title="Reset all changes for all images"
          >
            <RotateCcw size={14} /> Reset All Boxes
          </button>

          {totalPages > 1 && (
            <div className="flex items-center gap-2 ml-auto">
                <p className="text-xs text-stone-500 font-medium mr-4">
                    {startIndex + 1}-{Math.min(endIndex, totalCount)} of {totalCount}
                </p>
                <button 
                  onClick={() => onPageChange('prev')} 
                  disabled={currentPage === 1} 
                  className="p-2 rounded-lg border border-stone-300 bg-white text-stone-600 hover:bg-emerald-50 hover:border-emerald-300 hover:text-emerald-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
                >
                  <ChevronLeft size={16} />
                </button>
                
                <span className="text-sm font-bold text-emerald-900 min-w-[80px] text-center">
                  Page {currentPage} / {totalPages}
                </span>
                
                <button 
                  onClick={() => onPageChange('next')} 
                  disabled={currentPage === totalPages} 
                  className="p-2 rounded-lg border border-stone-300 bg-white text-stone-600 hover:bg-emerald-50 hover:border-emerald-300 hover:text-emerald-700 disabled:opacity-40 disabled:cursor-not-allowed transition"
                >
                  <ChevronRight size={16} />
                </button>
            </div>
          )}
        </div>

      </div>
    </div>
  );
}
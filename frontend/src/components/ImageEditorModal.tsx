"use client";

import { 
  X, Loader2, MousePointer2, 
  RotateCcw, Save, Trash2,
  Plus, MousePointerClick
} from "lucide-react";

import { ImageFile } from "@/types";
import EditorSidebar from "@/components/image-editor/EditorSidebar";
import EditorCanvas from "@/components/image-editor/EditorCanvas";
import EditorToolbar from "@/components/image-editor/EditorToolbar";
import { useImageEditor } from "@/hooks/useImageEditor";

interface ImageEditorModalProps {
  selectedImage: ImageFile;
  onClose: () => void;
  onSave: (updatedImage: ImageFile) => void;
}

export default function ImageEditorModal({ 
  selectedImage, 
  onClose, 
  onSave 
}: ImageEditorModalProps) {
  
  // --- LOAD HOOK ---
  const {
    localBoxes, localScores, localEliminated, localModified, localIsManual,
    activeBoxIndex, suggestions, isMergeMode, mergeSelection, isDrawingMode,
    currentDrawingBox, imgNaturalSize, isSaving, imgElementRef,
    // Setters needed for direct UI manipulation
    setActiveBoxIndex, setIsDrawingMode, setIsMergeMode, setMergeSelection, setImgNaturalSize,
    // Handlers
    toggleBox, handleApplySuggestion, handleMergeSelected, handleBatchDelete,
    handleEliminateAll, toggleElimination, handleResizeStart,
    handleContainerMouseMove, handleContainerMouseDown, handleContainerMouseUp,
    handleDiscardChanges, handleSave
  } = useImageEditor(selectedImage, onSave);

  return (
    <div 
        className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-stone-900/60 backdrop-blur-sm animate-in fade-in duration-200"
        onMouseUp={handleContainerMouseUp}
    >
      <div className="bg-white rounded-2xl shadow-2xl max-w-5xl w-full max-h-[90vh] flex flex-col overflow-hidden border border-stone-200">
        
        {/* HEADER */}
        <div className="p-4 border-b border-stone-100 flex justify-between items-center bg-[#F6F4EF]">
          <div className="flex flex-col">
            <h3 className="font-bold text-stone-700 truncate pr-4 text-lg">{selectedImage.name}</h3>
            <span className="text-xs text-stone-400 flex items-center gap-1">
              {isMergeMode ? (
                 <><MousePointerClick size={12} /> Bulk Edit Mode Active</>
              ) : isDrawingMode ? (
                 <><Plus size={12} /> Drawing Mode Active</>
              ) : (
                 <><MousePointer2 size={12} /> Editing Mode Enabled</>
              )}
            </span>
          </div>
          <button onClick={onClose} className="p-2 hover:bg-stone-200 rounded-full text-stone-500 transition"><X size={20} /></button>
        </div>
        
        {/* BODY */}
        <div className="flex-1 overflow-y-auto bg-stone-50 p-6 flex flex-col lg:flex-row gap-6 relative">
          
          {isSaving && (
            <div className="absolute inset-0 bg-white/70 backdrop-blur-sm z-50 flex flex-col items-center justify-center text-emerald-700">
              <Loader2 size={48} className="animate-spin mb-2" />
              <span className="font-bold text-lg">Saving changes...</span>
            </div>
          )}

          {/* LEFT: IMAGE & TOOLS */}
          <div className="flex-1 flex flex-col gap-4">
            
            {/* CANVAS */}
            <EditorCanvas 
              ref={imgElementRef}
              imageUrl={selectedImage.url}
              naturalSize={imgNaturalSize}
              boxes={localBoxes}
              scores={localScores}
              eliminated={localEliminated}
              isManual={localIsManual}
              activeBoxIndex={activeBoxIndex}
              mergeSelection={mergeSelection}
              isMergeMode={isMergeMode}
              isDrawingMode={isDrawingMode}
              currentDrawingBox={currentDrawingBox}
              
              onImgLoad={(e) => {
                const img = e.target as HTMLImageElement;
                setImgNaturalSize({ w: img.naturalWidth, h: img.naturalHeight });
              }}
              onBoxClick={toggleBox}
              onToggleElimination={toggleElimination}
              onResizeStart={handleResizeStart}
              onMouseMove={handleContainerMouseMove}
              onMouseDown={handleContainerMouseDown}
            />

            {/* TOOLBAR */}
            <EditorToolbar 
              isMergeMode={isMergeMode}
              isDrawingMode={isDrawingMode}
              mergeSelectionCount={mergeSelection.length}
              onStartDrawing={() => { setIsDrawingMode(true); setActiveBoxIndex(null); }}
              onStartMerge={() => { setIsMergeMode(true); setActiveBoxIndex(null); }}
              onCancelMode={() => { setIsDrawingMode(false); setIsMergeMode(false); setMergeSelection([]); }}
              onBatchDelete={handleBatchDelete}
              onMergeSelected={handleMergeSelected}
            />
          </div>

          {/* RIGHT: SIDEBAR */}
          <EditorSidebar 
            boxes={localBoxes}
            scores={localScores}
            eliminated={localEliminated}
            modified={localModified}
            isManual={localIsManual}
            activeBoxIndex={activeBoxIndex}
            mergeSelection={mergeSelection}
            isMergeMode={isMergeMode}
            suggestions={suggestions}
            onToggleBox={toggleBox}
            onToggleElimination={toggleElimination}
            onApplySuggestion={handleApplySuggestion}
            onClearSelection={() => setActiveBoxIndex(null)}
            onStartDrawing={() => { setIsDrawingMode(true); setActiveBoxIndex(null); }}
          />

        </div>

        {/* FOOTER */}
        <div className="p-4 border-t border-stone-100 bg-white flex justify-between items-center gap-4">
          <span className="text-xs text-stone-400 italic hidden md:inline">Unsaved changes will be lost upon closing.</span>
          <div className="flex gap-3 ml-auto">
            <button 
              onClick={handleEliminateAll}
              disabled={isSaving}
              className="px-4 py-2 bg-red-50 hover:bg-red-100 text-red-600 rounded-lg font-medium transition flex items-center gap-2 text-sm border border-red-200 hover:border-red-300 disabled:opacity-50"
            >
              <Trash2 size={16} /> Eliminate All
            </button>

            <button onClick={handleDiscardChanges} disabled={isSaving} className="px-4 py-2 bg-stone-100 hover:bg-stone-200 text-stone-600 rounded-lg font-medium transition flex items-center gap-2 text-sm disabled:opacity-50">
              <RotateCcw size={16} /> {activeBoxIndex !== null ? "Reset Selected" : "Reset All"}
            </button>
            <button onClick={handleSave} disabled={isSaving} className="px-6 py-2 bg-emerald-600 hover:bg-emerald-700 text-white rounded-lg font-medium transition flex items-center gap-2 text-sm shadow-sm disabled:opacity-70 disabled:cursor-not-allowed">
              {isSaving ? <><Loader2 size={16} className="animate-spin"/> Saving...</> : <><Save size={16} /> Save Changes</>}
            </button>
          </div>
        </div>

      </div>
    </div>
  );
}
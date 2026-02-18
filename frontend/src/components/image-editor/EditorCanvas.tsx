import { forwardRef } from "react";
import { Trash2, RefreshCcw, CheckSquare } from "lucide-react";
import { getScoreColor } from "@/utils/editorUtils";

interface EditorCanvasProps {
  imageUrl: string;
  naturalSize: { w: number; h: number } | null;
  boxes: number[][];
  scores: number[];
  eliminated: boolean[];
  isManual: boolean[];
  activeBoxIndex: number | null;
  mergeSelection: number[];
  isMergeMode: boolean;
  isDrawingMode: boolean;
  currentDrawingBox: number[] | null;
  
  // Events
  onImgLoad: (e: React.SyntheticEvent<HTMLImageElement>) => void;
  onBoxClick: (index: number) => void;
  onToggleElimination: (e: React.MouseEvent, index: number) => void;
  onResizeStart: (e: React.MouseEvent, index: number, handle: string) => void;
  
  // Mouse Interaction (Passed from parent's complex logic)
  onMouseMove: (e: React.MouseEvent) => void;
  onMouseDown: (e: React.MouseEvent) => void;
}

const EditorCanvas = forwardRef<HTMLImageElement, EditorCanvasProps>(({
  imageUrl,
  naturalSize,
  boxes,
  scores,
  eliminated,
  isManual,
  activeBoxIndex,
  mergeSelection,
  isMergeMode,
  isDrawingMode,
  currentDrawingBox,
  onImgLoad,
  onBoxClick,
  onToggleElimination,
  onResizeStart,
  onMouseMove,
  onMouseDown
}, ref) => {

  return (
    <div 
      className={`relative w-full h-auto min-h-[300px] flex justify-center items-center rounded-lg overflow-hidden border-2 select-none transition-colors
        ${isMergeMode ? 'bg-indigo-50/50 border-indigo-200' : ''}
        ${isDrawingMode ? 'cursor-crosshair bg-stone-100 border-blue-300' : ''}
        ${!isMergeMode && !isDrawingMode ? 'bg-stone-200/50 border-stone-200' : ''}
      `}
      onMouseMove={onMouseMove}
      onMouseDown={onMouseDown}
    >
       <div className="relative inline-block">
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img 
            ref={ref}
            src={imageUrl} 
            alt="Analyzed"
            className="max-h-[60vh] max-w-full h-auto w-auto object-contain block pointer-events-none"
            onLoad={onImgLoad}
          />
          
          {/* VISUAL PREVIEW OF DRAWING BOX */}
          {currentDrawingBox && naturalSize && (
              <div className="absolute border-2 border-blue-500 bg-blue-500/20 z-50 pointer-events-none"
                   style={{
                       left: `${(currentDrawingBox[0] / naturalSize.w) * 100}%`,
                       top: `${(currentDrawingBox[1] / naturalSize.h) * 100}%`,
                       width: `${((currentDrawingBox[2] - currentDrawingBox[0]) / naturalSize.w) * 100}%`,
                       height: `${((currentDrawingBox[3] - currentDrawingBox[1]) / naturalSize.h) * 100}%`
                   }}
              />
          )}

          {naturalSize && boxes.map((box, idx) => {
            if (box.length < 4) return null;
            const isActive = activeBoxIndex === idx;
            const isElim = eliminated[idx];
            const isMan = isManual[idx];
            const isSelectedForMerge = mergeSelection.includes(idx);

            // Visibility Logic
            const isVisible = activeBoxIndex === null ? !isElim : isActive;
            
            if (!isVisible && !isMergeMode && !isDrawingMode) return null;
            if (isMergeMode && isElim) return null;

            const score = scores[idx] || 0;
            const color = isElim ? '#ef4444' : getScoreColor(score, isMan);
            
            const left = (box[0] / naturalSize.w) * 100;
            const top = (box[1] / naturalSize.h) * 100;
            const width = ((box[2] - box[0]) / naturalSize.w) * 100;
            const height = ((box[3] - box[1]) / naturalSize.h) * 100;

            return (
              <div 
                key={idx}
                onClick={(e) => {
                    e.stopPropagation();
                    if (!isDrawingMode && !isElim) onBoxClick(idx);
                }}
                className={`absolute flex items-start justify-start transition-none
                  ${isMergeMode && isSelectedForMerge ? 'z-40 ring-2 ring-indigo-500 bg-indigo-500/20' : ''}
                  ${isActive && !isElim && !isMergeMode && !isDrawingMode ? 'z-30 ring-2 ring-white shadow-lg border-2' : 'hover:z-20'} 
                  ${isElim ? 'z-10 border border-dashed border-red-500 bg-red-500/10' : 'border-2'}
                  ${isMergeMode && !isSelectedForMerge ? 'opacity-50 hover:opacity-100' : ''}
                  ${isDrawingMode ? 'pointer-events-none opacity-40' : ''}
                `}
                style={{
                  left: `${left}%`, top: `${top}%`, width: `${width}%`, height: `${height}%`,
                  borderColor: isMergeMode && isSelectedForMerge ? '#6366f1' : color, 
                  backgroundColor: isActive ? `${color}33` : (isElim ? 'rgba(239, 68, 68, 0.1)' : undefined), 
                  cursor: isElim ? 'not-allowed' : 'pointer'
                }}
              >
                {!isMergeMode && !isElim && !isDrawingMode && (
                    <div className="absolute left-0 -top-5 flex shadow-sm pointer-events-auto">
                        <span className="text-[10px] font-bold text-white px-1.5 py-0.5 rounded-l-sm whitespace-nowrap" style={{ backgroundColor: color }}>
                            {isMan ? 'M' : `#${idx + 1}`}
                        </span>
                        {isActive && (
                            <button onClick={(e) => onToggleElimination(e, idx)} className="bg-stone-800 text-white px-1.5 py-0.5 hover:bg-red-600 transition rounded-r-sm"><Trash2 size={10} /></button>
                        )}
                    </div>
                )}
                
                {isMergeMode && (
                     <div className={`absolute -top-3 -right-3 p-1 rounded-full shadow-sm ${isSelectedForMerge ? 'bg-indigo-600 text-white' : 'bg-white text-stone-300'}`}>
                        <CheckSquare size={14} />
                     </div>
                )}

                {isElim && !isDrawingMode && (
                    <>
                        <svg className="absolute inset-0 w-full h-full pointer-events-none opacity-50">
                            <line x1="0" y1="0" x2="100%" y2="100%" stroke="#ef4444" strokeWidth="2" />
                            <line x1="100%" y1="0" x2="0" y2="100%" stroke="#ef4444" strokeWidth="2" />
                        </svg>
                        <div className="absolute inset-0 flex items-center justify-center z-50 pointer-events-auto">
                            <button onClick={(e) => onToggleElimination(e, idx)} className="bg-white text-emerald-600 p-2 rounded-full shadow-md hover:bg-emerald-50"><RefreshCcw size={20} /></button>
                        </div>
                    </>
                )}
                
                {/* RESIZE HANDLES */}
                {isActive && !isElim && !isMergeMode && !isDrawingMode && (
                  <>
                    {['nw', 'ne', 'sw', 'se'].map(h => (
                        <div key={h} className={`absolute w-3 h-3 bg-white border-2 border-emerald-600 rounded-full cursor-${h}-resize z-40 pointer-events-auto ${h.includes('n')?'-top-1.5':'-bottom-1.5'} ${h.includes('w')?'-left-1.5':'-right-1.5'}`} onMouseDown={(e) => onResizeStart(e, idx, h)} />
                    ))}
                  </>
                )}
              </div>
            );
          })}
       </div>
    </div>
  );
});

EditorCanvas.displayName = "EditorCanvas";
export default EditorCanvas;
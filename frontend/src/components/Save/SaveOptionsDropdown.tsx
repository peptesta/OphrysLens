"use client";

import { useRef, useEffect, useState, ReactNode, ChangeEvent } from "react";
import { 
  Save, Loader2, ChevronDown, Sparkles, 
  Layers, ArrowLeft, Percent, Maximize2 
} from "lucide-react";
import { SaveStrategy } from "@/utils/saveStrategies";

interface SaveOptionsDropdownProps {
  isSaving: boolean;
  isOpen: boolean;
  onToggle: () => void;
  onSelect: (strategy: SaveStrategy, threshold?: number, resize?: { width: number; height: number }) => void; 
  onClose: () => void;
}

interface OptionButtonProps {
  onClick: () => void;
  icon: ReactNode;
  title: string;
  desc: string;
  hasSubmenu?: boolean;
}

export default function SaveOptionsDropdown({ isSaving, isOpen, onToggle, onSelect, onClose }: SaveOptionsDropdownProps) {
  const dropdownRef = useRef<HTMLDivElement>(null);
  const [view, setView] = useState<'main' | 'threshold_selection'>('main');
  const [activeStrategy, setActiveStrategy] = useState<SaveStrategy>('global_best');
  
  const [inputValue, setInputValue] = useState<string>("0");
  const [useResize, setUseResize] = useState(false);
  const [dimensions, setDimensions] = useState({ width: 256, height: 512 });

  // Gestione chiusura cliccando fuori (usa onClose)
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (isOpen && dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isOpen, onClose]);

  useEffect(() => {
    if (!isOpen) {
      setTimeout(() => { setView('main'); setUseResize(false); }, 200);
    }
  }, [isOpen]);

  const handleThresholdChange = (e: ChangeEvent<HTMLInputElement>) => {
    const val = e.target.value;
    if (val === "") return setInputValue("");
    const parsed = Math.max(0, Math.min(100, parseInt(val) || 0));
    setInputValue(parsed.toString());
  };

  const getCurrentScore = () => (inputValue === "" ? 0 : parseInt(inputValue));

  const handleFinalSave = (withThreshold: boolean) => {
    const threshold = withThreshold ? getCurrentScore() : undefined;
    const resizeObj = useResize ? dimensions : undefined;
    onSelect(activeStrategy, threshold, resizeObj);
  };

  return (
    <div className="relative" ref={dropdownRef}>
      <button 
        onClick={onToggle} 
        disabled={isSaving} 
        className="flex items-center gap-2 bg-emerald-600 hover:bg-emerald-700 text-white px-5 py-2.5 rounded-lg font-bold shadow-md min-w-[200px] justify-between transition-all disabled:opacity-70"
      >
        {isSaving ? <Loader2 size={18} className="animate-spin mx-auto" /> : <><Save size={18} /> Save Options <ChevronDown size={16} /></>}
      </button>

      {isOpen && !isSaving && (
        <div className="absolute bottom-full right-0 mb-2 w-80 bg-white rounded-xl shadow-2xl border border-stone-100 overflow-hidden z-20 p-1 animate-in fade-in slide-in-from-bottom-2 duration-200">
          {view === 'main' ? (
            <div className="space-y-1">
              <OptionButton 
                onClick={() => { setActiveStrategy('global_best'); setView('threshold_selection'); }} 
                icon={<Sparkles size={18} className="text-emerald-600"/>} 
                title="Save Best / Filtered" 
                desc="Highest score or > X% for ALL." 
                hasSubmenu 
              />
              <OptionButton 
                onClick={() => onSelect('global_all')} 
                icon={<Layers size={18} className="text-blue-600"/>} 
                title="Save All Crops" 
                desc="Every detection for ALL images." 
              />
            </div>
          ) : (
            <div className="p-2 space-y-4">
              <button onClick={() => setView('main')} className="flex items-center gap-2 text-[10px] font-bold text-stone-400 hover:text-stone-600 uppercase">
                <ArrowLeft size={14}/> Back
              </button>
              
              {/* SEZIONE RESIZE */}
              <div className="bg-stone-50 p-3 rounded-lg border border-stone-200">
                <div className="flex items-center justify-between mb-2">
                  <label className="text-xs font-bold flex items-center gap-2 text-stone-700">
                    <Maximize2 size={14}/> Fixed Resize
                  </label>
                  <input type="checkbox" checked={useResize} onChange={(e) => setUseResize(e.target.checked)} className="accent-emerald-600" />
                </div>
                {useResize && (
                  <div className="flex gap-2 items-center">
                    <input type="number" value={dimensions.width} onChange={(e) => setDimensions({...dimensions, width: parseInt(e.target.value)||0})} className="w-full p-1.5 text-xs border rounded text-center font-mono" placeholder="W" />
                    <span className="text-stone-400">×</span>
                    <input type="number" value={dimensions.height} onChange={(e) => setDimensions({...dimensions, height: parseInt(e.target.value)||0})} className="w-full p-1.5 text-xs border rounded text-center font-mono" placeholder="H" />
                  </div>
                )}
              </div>

              {/* SEZIONE FILTRO SCORE */}
              <div className="bg-white p-3 rounded-lg border border-emerald-100 shadow-sm">
                <label className="text-xs font-bold text-stone-700 mb-2 block flex items-center gap-2">
                  <Percent size={14} className="text-emerald-600"/> Threshold Filter
                </label>
                <div className="flex gap-2">
                  <input type="number" value={inputValue} onChange={handleThresholdChange} className="w-16 p-2 text-sm border rounded text-center font-mono" />
                  <button onClick={() => handleFinalSave(true)} className="flex-1 bg-emerald-600 text-white text-xs font-bold rounded hover:bg-emerald-700 transition-colors">
                    Save &gt; {getCurrentScore()}%
                  </button>
                </div>
              </div>

              <button onClick={() => handleFinalSave(false)} className="w-full py-2.5 border border-stone-200 rounded-lg text-sm font-bold text-stone-600 hover:bg-stone-50 transition-colors">
                Save Only Absolute Best
              </button>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function OptionButton({ onClick, icon, title, desc, hasSubmenu }: OptionButtonProps) {
  return (
    <button onClick={onClick} className="w-full text-left px-4 py-3 hover:bg-stone-50 rounded-lg flex items-center justify-between group transition-colors">
      <div className="flex items-center gap-3">
        {icon}
        <div>
          <span className="block text-sm font-bold text-stone-800">{title}</span>
          <span className="block text-[10px] text-stone-400">{desc}</span>
        </div>
      </div>
      {hasSubmenu && <ChevronDown size={14} className="-rotate-90 text-stone-300" />}
    </button>
  );
}
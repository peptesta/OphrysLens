"use client";

import { useState, useRef, useEffect } from "react";

const CLASS_NAMES = [
  'O. exaltata', 
  'O. garganica', 
  'O. incubacea', 
  'O. majellensis', 
  'O. sphegodes', 
  'O. sphegodes_Palena'
];

interface ProbabilityDropdownProps {
  probs?: Record<string, number> | number[]; 
}

export default function ProbabilityDropdown({ probs }: ProbabilityDropdownProps) {
  const [isOpen, setIsOpen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  // Close dropdown if clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (containerRef.current && !containerRef.current.contains(event.target as Node)) {
        setIsOpen(false);
      }
    }
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  if (!probs) return null;

  // Data Normalization
  let entries: [string, number][] = [];
  try {
    if (Array.isArray(probs)) {
      if (probs.length === 0) return null;
      entries = probs.map((score, index) => [
        CLASS_NAMES[index] || `Class ${index + 1}`, 
        Number(score)
      ]);
    } else {
      if (Object.keys(probs).length === 0) return null;
      entries = Object.entries(probs).map(([k, v]) => [k, Number(v)]);
    }
  } catch (e) {
    console.log("Error normalizing probabilities:", e);
    return null;
  }

  // Sort descending
  const sortedProbs = entries.sort(([, a], [, b]) => b - a);

  return (
    <div className="w-full relative" ref={containerRef}>
      {/* Toggle Button */}
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="flex items-center gap-2 text-xs font-bold text-emerald-700 hover:text-emerald-800 transition-colors uppercase tracking-wider focus:outline-none select-none w-full text-left p-1 rounded hover:bg-emerald-50"
      >
        <span>{isOpen ? "Hide Breakdown" : "View All Probabilities"}</span>
        <span className={`transform transition-transform duration-200 ${isOpen ? "rotate-180" : ""}`}>
          ▼
        </span>
      </button>

      {/* Floating Content (Absolute Position) */}
      {isOpen && (
        <div className="absolute top-full left-0 w-full z-50 mt-2 transform origin-top animate-fade-in-down">
          <div className="bg-white rounded-xl border border-stone-200 shadow-xl overflow-hidden ring-1 ring-black ring-opacity-5">
            <div className="max-h-64 overflow-y-auto custom-scrollbar p-1">
              {sortedProbs.map(([label, score]) => (
                <div key={label} className="flex justify-between items-center p-2 hover:bg-stone-50 rounded-lg transition-colors group">
                  {/* Class Name */}
                  <span className="text-sm text-stone-600 font-medium group-hover:text-stone-900 truncate pr-2 min-w-0 flex-1" title={label}>
                    {label}
                  </span>
                  
                  {/* Score & Bar */}
                  <div className="flex items-center gap-2 flex-shrink-0">
                    {/* Width increased from w-16 to w-28 */}
                    <div className="w-28 h-1.5 bg-stone-100 rounded-full overflow-hidden border border-stone-100">
                      <div 
                        className={`h-full rounded-full ${score > 50 ? 'bg-emerald-500' : 'bg-stone-400'}`}
                        style={{ width: `${isNaN(score) ? 0 : score}%` }} 
                      />
                    </div>
                    <span className="text-stone-900 font-mono text-xs w-10 text-right font-bold">
                      {isNaN(score) ? "0.0" : score.toFixed(1)}%
                    </span>
                  </div>
                </div>
              ))}
            </div>
            {/* Deleted footer gradient div here */}
          </div>
        </div>
      )}
    </div>
  );
}
import { ApiResponse } from "@/types";
import CompareView from "./CompareView";
import SingleView from "./SingleView";

interface ResultsDisplayProps {
  result: ApiResponse;
  preview: string; 
  currentCropMode: string;
  analyzedCropMode: string | null;
  currentStrategy: string;      // <--- New Prop
  analyzedStrategy: string | null; // <--- New Prop
  useGpu: boolean;
}

export default function ResultsDisplay({ 
  result, 
  preview, 
  currentCropMode, 
  analyzedCropMode,
  currentStrategy,
  analyzedStrategy,
  useGpu 
}: ResultsDisplayProps) {
  
  // Format Strategy Name for UI
  const getStrategyName = (strat: string | null) => {
    if (strat === "1vsall") return "1-vs-All Ensemble";
    return "Standard 6-Class Model";
  };
  const displayStrategyName = getStrategyName(analyzedStrategy);

  // --- LOGIC 1: Compare Mode ---
  // If we are in compare mode, we render. CompareView handles partial data internally.
  // BUT: If the Strategy changed, we force a re-run prompt even for Compare view, 
  // because comparing "Standard Left" vs "1vsAll Right" is confusing/invalid.
  const strategyChanged = currentStrategy !== analyzedStrategy;

  if (strategyChanged) {
     return (
      <div className="animate-fade-in max-w-4xl mx-auto mt-10 p-12 text-center bg-white rounded-2xl shadow-sm border border-stone-200">
        <span className="text-4xl block mb-4">⚙️</span>
        <h3 className="text-xl font-bold text-emerald-900 mb-2">Strategy Changed</h3>
        <p className="text-stone-500">
          You switched to <span className="font-bold text-stone-700">{getStrategyName(currentStrategy)}</span>.
          <br/>
          Click <span className="font-bold text-emerald-600">Identify Species</span> to update results.
        </p>
      </div>
    );
  }

  if (currentCropMode === "compare") {
    return (
      <CompareView 
        result={result} 
        preview={preview} 
        analyzedMode={analyzedCropMode} 
        strategyName={displayStrategyName} // <--- Pass Name
      />
    );
  }

  // --- LOGIC 2: Single Mode Mismatch ---
  if (currentCropMode !== analyzedCropMode) {
    return (
      <div className="animate-fade-in max-w-4xl mx-auto mt-10 p-12 text-center bg-white rounded-2xl shadow-sm border border-stone-200">
        <span className="text-4xl block mb-4">🔄</span>
        <h3 className="text-xl font-bold text-emerald-900 mb-2">View Changed</h3>
        <p className="text-stone-500">
          You switched to <span className="font-bold text-stone-700 uppercase">{currentCropMode}</span> view.
          <br/>
          Click <span className="font-bold text-emerald-600">Identify Species</span> to see the new results.
        </p>
      </div>
    );
  }

  // --- LOGIC 3: Single Mode Matches ---
  return (
    <SingleView 
      result={result} 
      preview={preview} 
      mode={currentCropMode} 
      useGpu={useGpu} 
      strategyName={displayStrategyName} // <--- Pass Name
    />
  );
}
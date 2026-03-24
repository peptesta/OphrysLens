'use client';

import { useState } from "react";
import Image from "next/image";
import { ApiResponse } from "@/types";
import PredictionCard from "@/components/PredictionCard";
import ProbabilityDropdown from "@/components/ProbabilityDropdown";
import { useImageMetadata } from "@/hooks/useImageMetadata";

interface CompareViewProps {
  result: ApiResponse;
  preview: string;
  analyzedMode: string | null;
  strategyName: string;
}

function XaiCombinedDisplay({ 
  combinedSrc, 
  title, 
  onImageClick 
}: { 
  combinedSrc?: string | null, 
  title: string, 
  onImageClick: (src: string, label: string) => void 
}) {
  if (!combinedSrc) return null;
  return (
    <div className="mt-6 pt-6 border-t border-stone-100">
      <h4 className="text-[10px] font-bold text-emerald-800 uppercase mb-4 tracking-widest flex items-center gap-2">
        🧠 {title} Analysis (Occlusion & IG)
      </h4>
      <div 
        onClick={() => onImageClick(`data:image/png;base64,${combinedSrc}`, `${title} - Visual Explanation`)}
        className="relative h-64 bg-white rounded-lg border border-stone-200 overflow-hidden group cursor-zoom-in hover:border-emerald-500 transition-all shadow-sm"
      >
        <Image 
          src={`data:image/png;base64,${combinedSrc}`} 
          alt="Combined XAI" 
          fill 
          className="object-contain p-1" 
          unoptimized 
        />
      </div>
    </div>
  );
}

function EmptyStateCard({ title }: { title: string }) {
  return (
    <div className="h-full flex flex-col items-center justify-center p-8 text-center bg-stone-50 rounded-xl border-2 border-dashed border-stone-200 min-h-[400px]">
      <span className="text-3xl mb-3 opacity-40">⏳</span>
      <h3 className="font-bold text-stone-600 mb-1">{title}</h3>
      <p className="text-sm text-stone-500">Click Identify Species to see results.</p>
    </div>
  );
}

// --- COMPONENTE PRINCIPALE ---
export default function CompareView({ result, preview, analyzedMode, strategyName }: CompareViewProps) {
  const [showMetadata, setShowMetadata] = useState(false);
  const [fullscreenImg, setFullscreenImg] = useState<{src: string, label: string} | null>(null);
  
  const { metadata, address, loading: metaLoading } = useImageMetadata(preview);

  const showLeft  = analyzedMode === 'integrated' || analyzedMode === 'compare';
  const showRight = analyzedMode === 'external'   || analyzedMode === 'compare';

  const hasGps = metadata?.latitude !== undefined && metadata?.longitude !== undefined;
  const lat = metadata?.latitude;
  const lon = metadata?.longitude;

  const leftData = {
    class: result.predicted_class,
    confidence: result.confidence,
    probs: result.all_classes_probs,
    ig: result.integrated_gradients,
    occ: result.occlusion,
    combined: result.explanation_combined
  };

  const rightData = {
    class:
      analyzedMode === "external"
        ? result.predicted_class
        : result.predicted_class_cropped,

    confidence:
      analyzedMode === "external"
        ? result.confidence
        : result.confidence_cropped || 0,

    probs:
      analyzedMode === "external"
        ? result.all_classes_probs
        : result.all_classes_probs_cropped,

    ig:
      analyzedMode === "external"
        ? result.integrated_gradients
        : result.integrated_gradients_cropped,

    occ:
      analyzedMode === "external"
        ? result.occlusion
        : result.occlusion_cropped,

    crop: result.image_cropped,
    combined: result.explanation_combined_cropped
  };

  const handleImageClick = (src: string, label: string) => {
    setFullscreenImg({ src, label });
  };

  return (
    <div className="space-y-8 relative">
      
      {/* --- LIGHTBOX (FULLSCREEN MODAL) --- */}
      {fullscreenImg && (
        <div 
          className="fixed inset-0 z-[999] bg-stone-950/90 backdrop-blur-md flex flex-col items-center justify-center p-6 md:p-12 cursor-pointer transition-all"
          onClick={() => setFullscreenImg(null)}
        >
          <button className="absolute top-8 right-8 text-white/50 text-5xl hover:text-white transition-colors">&times;</button>
          <div className="relative w-full h-full max-w-6xl flex flex-col items-center justify-center">
            <p className="text-emerald-400 font-bold uppercase tracking-[0.2em] text-xs mb-6 bg-emerald-950/50 px-4 py-2 rounded-full border border-emerald-800/50">
                {fullscreenImg.label}
            </p>
            <div className="relative w-full h-[80vh]">
                <Image 
                    src={fullscreenImg.src} 
                    alt="Fullscreen" 
                    fill 
                    className="object-contain" 
                    unoptimized
                />
            </div>
            <p className="text-white/30 text-[10px] mt-8 uppercase tracking-widest font-bold">Click anywhere to close</p>
          </div>
        </div>
      )}

      {/* Header Strategy */}
      <div className="flex flex-col items-center gap-4">
         <span className="bg-emerald-100 text-emerald-800 text-xs font-bold px-4 py-1.5 rounded-full border border-emerald-200 shadow-sm">
            Current Strategy: {strategyName}
         </span>
      </div>

      {/* Pannelli di Comparazione */}
      <div className="grid grid-cols-1 xl:grid-cols-2 gap-8">
        
        {/* --- LEFT PANEL: Integrated --- */}
        <div className="bg-white p-6 rounded-2xl shadow-md border border-stone-200 flex flex-col">
          {showLeft ? (
            <>
              <div className="mb-6 pb-4 border-b border-stone-100">
                <PredictionCard title="Integrated Mode (Original)" className={leftData.class} confidence={leftData.confidence} />
                <div className="mt-4"><ProbabilityDropdown probs={leftData.probs} /></div>
              </div>
              <div 
                className="relative w-full h-[400px] bg-stone-100 rounded-lg overflow-hidden border border-stone-200 cursor-zoom-in group"
                onClick={() => handleImageClick(preview, "Original Input Image")}
              >
                 <Image src={preview} alt="Original" fill className="object-contain group-hover:scale-[1.02] transition-transform duration-500" unoptimized />
              </div>
              <XaiCombinedDisplay 
                combinedSrc={leftData.combined} 
                title="Original" 
                onImageClick={handleImageClick} 
              /> 
           </>
          ) : (
            <EmptyStateCard title="Integrated Results Pending" />
          )}
        </div>

        {/* --- RIGHT PANEL: External --- */}
        <div className="bg-white p-6 rounded-2xl shadow-md border border-stone-200 flex flex-col">
          {showRight ? (
            rightData.class ? (
              <>
                <div className="mb-6 pb-4 border-b border-stone-100">
                  <PredictionCard title="External Mode (Smart Crop)" className={rightData.class} confidence={rightData.confidence} />
                  <div className="mt-4"><ProbabilityDropdown probs={rightData.probs} /></div>
                </div>
                <div 
                    className="relative w-full h-[400px] bg-stone-100 rounded-lg overflow-hidden border border-stone-200 cursor-zoom-in group"
                    onClick={() => rightData.crop && handleImageClick(`data:image/jpeg;base64,${rightData.crop}`, "Focused Subject Crop")}
                >
                   {rightData.crop ? (
                      <Image src={`data:image/jpeg;base64,${rightData.crop}`} alt="Cropped" fill className="object-contain group-hover:scale-[1.02] transition-transform duration-500" unoptimized />
                   ) : (
                      <div className="h-full flex items-center justify-center text-stone-400 italic">No Crop Available</div>
                   )}
                </div>
                <XaiCombinedDisplay 
                  combinedSrc={rightData.combined} 
                  title="Original" 
                  onImageClick={handleImageClick} 
                />              </>
            ) : (
              <div className="h-full flex flex-col items-center justify-center p-8 bg-amber-50 rounded-xl border border-amber-100 min-h-[400px]">
                <span className="text-4xl mb-2">⚠️</span>
                <h3 className="font-bold text-amber-800">Crop Failed</h3>
                <p className="text-sm text-amber-600">Smart focus could not isolate the subject.</p>
              </div>
            )
          ) : (
            <EmptyStateCard title="External Results Pending" />
          )}
        </div>
      </div>

      {/* SEZIONE METADATI & GPS */}
      {(metadata || metaLoading) && (
        <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 bg-white p-6 rounded-2xl shadow-md border border-stone-200">
          <div className="lg:col-span-3">
            <h3 className="font-bold text-stone-800 flex items-center gap-2 text-sm uppercase mb-4">📍 Location Metadata</h3>
            {metaLoading ? (
               <div className="h-48 bg-stone-50 animate-pulse rounded-xl flex items-center justify-center text-stone-400 text-xs italic font-medium tracking-wide">Fetching GPS data...</div>
            ) : hasGps && lat && lon ? (
              <div className="space-y-4">
                <div className="bg-emerald-50 p-3 rounded-lg border border-emerald-100 flex justify-between items-center">
                   <div>
                    <span className="block text-[9px] uppercase text-emerald-600 font-bold mb-0.5">Detected Area</span>
                    <p className="text-emerald-900 font-semibold text-sm leading-tight">{address}</p>
                   </div>
                   <a href={`http://google.com/maps?q=${lat},${lon}`} target="_blank" rel="noopener noreferrer" className="bg-white p-2 rounded-full shadow-sm hover:scale-110 transition-transform">🌍</a>
                </div>
                <div className="w-full h-48 rounded-xl overflow-hidden border border-stone-200 shadow-inner">
                  <iframe width="100%" height="100%" frameBorder="0" src={`https://www.openstreetmap.org/export/embed.html?bbox=${lon-0.005}%2C${lat-0.005}%2C${lon+0.005}%2C${lat+0.005}&layer=mapnik&marker=${lat}%2C${lon}`}></iframe>
                </div>
              </div>
            ) : (
              <div className="h-48 bg-stone-50 rounded-xl flex items-center justify-center border border-dashed border-stone-200 text-stone-400 text-sm italic">GPS Coordinates not found in original file</div>
            )}
          </div>

          <div className="lg:col-span-2">
            <div className="flex justify-between items-center mb-4">
              <h3 className="font-bold text-stone-800 text-sm uppercase">ℹ️ Image Info</h3>
              <button onClick={() => setShowMetadata(!showMetadata)} className="text-xs font-bold text-pink-600 hover:text-pink-700 transition-colors underline">{showMetadata ? "Hide" : "Show All"}</button>
            </div>
            {metadata ? (
              <div className="space-y-3">
                <div className="grid grid-cols-1 gap-2 text-[11px]">
                  <div className="bg-stone-50 p-2.5 rounded-lg border border-stone-100 flex justify-between">
                    <span className="text-stone-400 font-bold uppercase text-[9px]">Camera</span>
                    <span className="text-stone-700 font-medium truncate max-w-[120px]">{metadata.Model || 'Generic'}</span>
                  </div>
                  <div className="bg-stone-50 p-2.5 rounded-lg border border-stone-100 flex justify-between">
                    <span className="text-stone-400 font-bold uppercase text-[9px]">Date</span>
                    <span className="text-stone-700 font-medium">{metadata.DateTimeOriginal ? new Date(metadata.DateTimeOriginal).toLocaleDateString() : 'N/A'}</span>
                  </div>
                </div>
                {showMetadata && (
                  <div className="mt-4 max-h-40 overflow-y-auto bg-stone-50 p-3 rounded-lg border border-stone-100 font-mono text-[9px] text-stone-600 custom-scrollbar">
                    <pre className="whitespace-pre-wrap">{JSON.stringify(metadata, null, 2)}</pre>
                  </div>
                )}
              </div>
            ) : <p className="text-xs text-stone-400 italic">Technical metadata unavailable.</p>}
          </div>
        </div>
      )}
    </div>
  );
}
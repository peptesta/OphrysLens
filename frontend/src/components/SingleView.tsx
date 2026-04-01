'use client';

import { useState } from "react";
import Image from "next/image";
import { ApiResponse } from "@/types";
import { getConfidenceColor } from "@/components/PredictionCard";
import ProbabilityDropdown from "@/components/ProbabilityDropdown";
import { useImageMetadata } from "@/hooks/useImageMetadata";

interface SingleViewProps {
  result: ApiResponse;
  preview: string;
  mode: string; 
  useGpu?: boolean;
  strategyName: string;
}

export default function SingleView({ result, preview, mode, strategyName }: SingleViewProps) {
  const [showMetadata, setShowMetadata] = useState(false);
  const [fullscreenImg, setFullscreenImg] = useState<{src: string, label: string} | null>(null);
  
  const isExternal = mode === "external";
  
  // Determina l'immagine da mostrare (originale o crop)
  const displayImageSrc = isExternal && result.image_cropped ? `data:image/jpeg;base64,${result.image_cropped}` : preview;
  const displayImageLabel = isExternal && result.image_cropped ? "Focused Crop Input" : "Original Input";

  // Estrazione metadati dall'immagine originale
  const { metadata, address, loading } = useImageMetadata(preview);

  const lat = metadata?.latitude;
  const lon = metadata?.longitude;
  const hasGps = lat !== undefined && lon !== undefined;

  const handleImageClick = (src: string, label: string) => {
    setFullscreenImg({ src, label });
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-5 gap-8 relative">
      
      {/* --- LIGHTBOX (FULLSCREEN MODAL) --- */}
      {fullscreenImg && (
        <div 
          className="fixed inset-0 z-[999] bg-stone-950/90 backdrop-blur-md flex flex-col items-center justify-center p-6 md:p-12 cursor-pointer"
          onClick={() => setFullscreenImg(null)}
        >
          <button className="absolute top-8 right-8 text-white/50 text-5xl hover:text-white transition-colors">×</button>
          <div className="relative w-full h-full max-w-6xl flex flex-col items-center justify-center">
            <p className="text-pink-400 font-bold uppercase tracking-[0.2em] text-xs mb-6 bg-pink-950/50 px-4 py-2 rounded-full border border-pink-800/50">
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
            <p className="text-white/30 text-[10px] mt-8 uppercase tracking-widest font-bold font-sans">Click anywhere to close</p>
          </div>
        </div>
      )}

      {/* --- COLONNA SINISTRA: Analisi e Dati --- */}
      <div className="lg:col-span-2 flex flex-col gap-6 order-2 lg:order-1">

        {/* 1. Card Risultato Principale */}
        <div className="bg-white p-8 rounded-2xl shadow-lg shadow-pink-100/50 border-l-8 border-pink-500">
          <div className="flex justify-between items-start mb-4">
            <span className="text-xs font-bold text-stone-400 uppercase tracking-widest">Identified Species</span>
            <span className="text-[10px] font-bold uppercase bg-stone-100 text-stone-600 px-2 py-1 rounded border border-stone-200">
              {strategyName}
            </span>
          </div>
          <h1 className={`text-4xl font-extrabold mt-3 mb-2 leading-tight ${getConfidenceColor(result.confidence)}`}>
            {result.predicted_class}
          </h1>
          <div className="flex items-center gap-2 mb-6">
            <span className={`inline-block w-3 h-3 rounded-full ${result.confidence > 80 ? 'bg-emerald-500' : 'bg-amber-500'}`}></span>
            <p className="text-stone-500 font-medium">{result.confidence.toFixed(2)}% Confidence Score</p>
          </div>
          
          <div className="pt-4 border-t border-stone-100">
            <h4 className="text-[10px] font-bold text-stone-400 uppercase mb-3 tracking-widest">Analysis Details</h4>
            <ProbabilityDropdown probs={result.all_classes_probs} />
          </div>
        </div>

        {/* 2. Sezione GPS & Mappa */}
        <div className="bg-white p-6 rounded-2xl shadow-md border border-stone-200 overflow-hidden">
          <h3 className="font-bold text-stone-800 flex items-center gap-2 text-sm uppercase mb-4">📍 Original Location</h3>
          {loading ? (
            <div className="h-48 bg-stone-100 animate-pulse rounded-xl flex items-center justify-center text-stone-400 text-xs italic">Extracting GPS...</div>
          ) : hasGps ? (
            <div className="space-y-4">
              <div className="bg-stone-50 p-3 rounded-lg border border-stone-100">
                <span className="block text-[9px] uppercase text-stone-400 font-bold mb-0.5">Detected Area</span>
                <p className="text-stone-800 font-semibold text-sm leading-tight">{address}</p>
              </div>
              <div className="w-full h-44 rounded-xl overflow-hidden border border-stone-200 shadow-inner relative">
                <iframe 
                  width="100%" height="100%" frameBorder="0" 
                  src={`https://www.openstreetmap.org/export/embed.html?bbox=${lon-0.005}%2C${lat-0.005}%2C${lon+0.005}%2C${lat+0.005}&layer=mapnik&marker=${lat}%2C${lon}`}
                ></iframe>
              </div>
            </div>
          ) : (
            <div className="h-24 bg-stone-50 rounded-xl flex flex-col items-center justify-center border border-dashed border-stone-200 text-stone-400 italic text-xs">Gps Coords not found</div>
          )}
        </div>

        {/* 3. Sezione Metadati EXIF */}
        <div className="bg-white p-6 rounded-2xl shadow-md border border-stone-200">
          <div className="flex justify-between items-center mb-4">
            <h3 className="font-bold text-stone-800 text-sm uppercase">ℹ️ Technical Meta</h3>
            <button onClick={() => setShowMetadata(!showMetadata)} className="text-xs font-bold text-pink-600 underline">
              {showMetadata ? "Hide" : "Show All"}
            </button>
          </div>
          {metadata ? (
            <div className="space-y-2 text-[11px]">
              <div className="grid grid-cols-2 gap-2">
                <div className="bg-stone-50 p-2 rounded">
                  <span className="block text-stone-400 font-bold uppercase text-[9px]">Model</span>
                  <span className="text-stone-700 font-medium truncate block">{metadata.Model || 'Unknown'}</span>
                </div>
                <div className="bg-stone-50 p-2 rounded">
                  <span className="block text-stone-400 font-bold uppercase text-[9px]">Date</span>
                  <span className="text-stone-700 font-medium truncate block">
                    {metadata.DateTimeOriginal ? new Date(metadata.DateTimeOriginal).toLocaleDateString() : 'N/A'}
                  </span>
                </div>
              </div>
              {showMetadata && (
                <div className="mt-4 max-h-48 overflow-y-auto bg-stone-50 p-3 rounded-lg border border-stone-100 font-mono text-[10px] custom-scrollbar">
                  <pre className="whitespace-pre-wrap">{JSON.stringify(metadata, null, 2)}</pre>
                </div>
              )}
            </div>
          ) : <p className="text-xs text-stone-400 italic font-medium">Meta non disponibili.</p>}
        </div>
      </div>

      {/* --- COLONNA DESTRA: Visualizzazioni --- */}
      <div className="lg:col-span-3 space-y-6 order-1 lg:order-2">
        
        {/* Preview Immagine Principale */}
        <div 
            className="bg-white p-4 rounded-xl shadow-md border border-stone-200 relative cursor-zoom-in group"
            onClick={() => handleImageClick(displayImageSrc, displayImageLabel)}
        >
          <h3 className="font-bold text-[10px] text-stone-400 uppercase mb-3 tracking-widest">{displayImageLabel}</h3>
          <div className="relative w-full h-[550px] bg-stone-100 rounded-lg overflow-hidden border border-stone-50">
            <Image src={displayImageSrc} alt="Input" fill className="object-contain group-hover:scale-[1.01] transition-transform duration-500" unoptimized />
          </div>
        </div>

        {(result.explanation_combined) && (
          <div className="bg-stone-50 p-6 rounded-xl border border-stone-200 shadow-inner">
            <h3 className="font-bold text-emerald-900 mb-6 flex items-center gap-2">
              🧠 Visual Explanation (Occlusion + Integrated Gradients)
            </h3>
            <div 
                className="relative w-full h-[450px] bg-white rounded-lg overflow-hidden shadow-sm border border-stone-200 cursor-zoom-in hover:border-emerald-500 transition-colors group"
                onClick={() => handleImageClick(`data:image/png;base64,${result.explanation_combined}`, "Visual Explanation Analysis")}
            >
              <Image 
                src={`data:image/png;base64,${result.explanation_combined}`} 
                alt="XAI Combined" 
                fill 
                className="object-contain p-4 group-hover:scale-[1.02] transition-transform" 
                unoptimized
              />
              <div className="absolute bottom-4 right-4 bg-stone-900/60 backdrop-blur-md text-white text-[9px] px-3 py-1 rounded-full font-bold uppercase tracking-tighter">
                Combined view
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
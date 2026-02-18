"use client";

import { useState, ChangeEvent } from "react";
import Image from "next/image";
import { ApiResponse } from "@/types";
import DashboardSidebar from "@/components/DashBoardSidebar";
import ResultsDisplay from "@/components/ResultDisplay";

const API_URL = "http://localhost:5000/inference";

export default function OrchidDashboard() {
  // --- STATE MANAGEMENT ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [resultCache, setResultCache] = useState<Record<string, ApiResponse>>({});

  // Settings
  const [modelStrategy, setModelStrategy] = useState("standard");
  const [cropMode, setCropMode] = useState("integrated");
  const [useGpu, setUseGpu] = useState(false);
  const [showOcclusion, setShowOcclusion] = useState(false);
  const [showIG, setShowIG] = useState(false);

  // TRACKING STATE: What generated the current result?
  const [analyzedMode, setAnalyzedMode] = useState<string | null>(null);
  const [analyzedStrategy, setAnalyzedStrategy] = useState<string | null>(null);

  // --- HELPERS ---
  const getCacheKey = (strat: string, crop: string, gpu: boolean, occ: boolean, ig: boolean) => {
    return `${strat}-${crop}-${gpu}-${occ}-${ig}`;
  };

  const checkCacheAndApply = (newStrat: string, newCrop: string, newGpu: boolean, newOcc: boolean, newIg: boolean) => {
    const key = getCacheKey(newStrat, newCrop, newGpu, newOcc, newIg);
    
    if (resultCache[key]) {
      setResult(resultCache[key]);
      setAnalyzedMode(newCrop);
      setAnalyzedStrategy(newStrat);
    } 
  };

  // --- EVENT HANDLERS ---
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const file = e.target.files[0];
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResult(null);
      setAnalyzedMode(null);
      setAnalyzedStrategy(null);
      setApiError(null);
      setResultCache({}); 
    }
  };

  const handleSetStrategy = (val: string) => {
    setModelStrategy(val);
    checkCacheAndApply(val, cropMode, useGpu, showOcclusion, showIG);
  };

  const handleSetCropMode = (val: string) => {
    setCropMode(val);
    checkCacheAndApply(modelStrategy, val, useGpu, showOcclusion, showIG);
  };

  const handleSetUseGpu = (val: boolean) => {
    setUseGpu(val);
    checkCacheAndApply(modelStrategy, cropMode, val, showOcclusion, showIG);
  };

  const handleSetOcc = (val: boolean) => {
    setShowOcclusion(val);
    checkCacheAndApply(modelStrategy, cropMode, useGpu, val, showIG);
  };

  const handleSetIG = (val: boolean) => {
    setShowIG(val);
    checkCacheAndApply(modelStrategy, cropMode, useGpu, showOcclusion, val);
  };

  // --- API CALL ---
  const handleAnalyze = async () => {
    if (!selectedFile) return;
    
    const currentKey = getCacheKey(modelStrategy, cropMode, useGpu, showOcclusion, showIG);
    
    if (resultCache[currentKey]) {
      setResult(resultCache[currentKey]);
      setAnalyzedMode(cropMode);
      setAnalyzedStrategy(modelStrategy);
      return;
    }

    setLoading(true);
    setApiError(null);

    let explainMethod = "none";
    if (useGpu) {
      if (showOcclusion && showIG) explainMethod = "both";
      else if (showOcclusion) explainMethod = "occlusion";
      else if (showIG) explainMethod = "integrated_gradients";
    }

    const formData = new FormData();
    formData.append("image", selectedFile);
    formData.append("model_strategy", modelStrategy);
    formData.append("crop_mode", cropMode);
    formData.append("explain_method", explainMethod);

    try {
      const res = await fetch(API_URL, { method: "POST", body: formData });
      if (!res.ok) throw new Error(`Server Error: ${res.statusText}`);
      const data: ApiResponse = await res.json();
      
      setResult(data);
      setAnalyzedMode(cropMode);
      setAnalyzedStrategy(modelStrategy);

      // Cache Logic
      const newCache = { ...resultCache };
      newCache[currentKey] = data;

      if (cropMode === "compare") {
        const intKey = getCacheKey(modelStrategy, "integrated", useGpu, showOcclusion, showIG);
        newCache[intKey] = data;

        const extKey = getCacheKey(modelStrategy, "external", useGpu, showOcclusion, showIG);
        const syntheticExternal: ApiResponse = {
           ...data,
           predicted_class: data.predicted_external_class || "", 
           confidence: data.confidence_external || 0,            
           all_classes_probs: data.all_classes_probs_external || [], 
           integrated_gradients: data.integrated_gradients_external,
           occlusion: data.occlusion_external,
        };
        newCache[extKey] = syntheticExternal;
      }

      setResultCache(newCache);

    } catch (err: unknown) {
      console.error(err);
      // Narrowing the error type to safely access the message
      const errorMessage = err instanceof Error ? err.message : "Failed to connect to backend";
      setApiError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen bg-[#F6F4EF] text-[#2A2F2C] flex flex-col md:flex-row font-sans overflow-hidden">
      <DashboardSidebar
        config={{ modelStrategy, cropMode, useGpu, showOcclusion, showIG }}
        setConfig={{ 
          setModelStrategy: handleSetStrategy, 
          setCropMode: handleSetCropMode, 
          setUseGpu: handleSetUseGpu, 
          setShowOcclusion: handleSetOcc, 
          setShowIG: handleSetIG 
        }}
        fileState={{ selectedFile, handleFileChange }}
        actionState={{ loading, handleAnalyze, apiError }}
      />

      <main className="flex-1 p-8 md:p-12 overflow-y-auto bg-pink-50 relative">
        <header className="mb-10">
          <h1 className="text-3xl md:text-4xl font-extrabold text-emerald-900 mb-2 flex items-center gap-3">
            <span className="text-pink-600">🌸</span> Inference for orchids
          </h1>
          <p className="text-stone-500 font-medium">Start By Choosing an Image</p>
        </header>

        {preview && !result && (
          <div className="max-w-4xl mx-auto border-2 border-dashed border-stone-300 rounded-2xl p-8 flex flex-col items-center justify-center bg-stone-50/50 text-center">
            <div className="relative w-full h-[500px] mb-4 p-2 bg-white rounded-xl shadow-sm border border-stone-100">
              <Image src={preview} alt="Upload" fill className="object-contain rounded-lg" unoptimized />
            </div>
            <p className="text-sm font-semibold text-stone-600 animate-pulse">Ready to analyze...</p>
          </div>
        )}

        {result && preview && (
          <ResultsDisplay 
            result={result}
            preview={preview}
            currentCropMode={cropMode}     
            analyzedCropMode={analyzedMode} 
            currentStrategy={modelStrategy}     
            analyzedStrategy={analyzedStrategy} 
            useGpu={useGpu}
          />
        )}
      </main>
    </div>
  );
}
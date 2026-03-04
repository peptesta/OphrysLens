"use client";

import { useState, ChangeEvent, useEffect } from "react";
import Image from "next/image";
import { ApiResponse } from "@/types";
import DashboardSidebar from "@/components/DashBoardSidebar";
import ResultsDisplay from "@/components/ResultDisplay";

// --- INTERFACCE PER I TIPI ---
interface ModelInfo {
  id: number;
  filename: string;
  label: string;
}

interface AvailableModelsResponse {
  "6class": ModelInfo[];
}

const BASE_API_URL = "http://localhost:5000/inference";

export default function OrchidDashboard() {
  // --- STATI PRINCIPALI ---
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<ApiResponse | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const [resultCache, setResultCache] = useState<Record<string, ApiResponse>>({});

  // --- STATI MODELLI (Tipizzato correttamente invece di string[]) ---
  const [availableModels, setAvailableModels] = useState<AvailableModelsResponse>({ "6class": [] });

  // --- STATI CONFIGURAZIONE ---
  const [modelStrategy, setModelStrategy] = useState("standard");
  const [cropMode, setCropMode] = useState("integrated");
  const [useGpu, setUseGpu] = useState(false);
  const [showOcclusion, setShowOcclusion] = useState(false);
  const [showIG, setShowIG] = useState(false);
  const [selectedModel6Class, setSelectedModel6Class] = useState<string>("");

  // --- STATI TRACKING ---
  const [analyzedMode, setAnalyzedMode] = useState<string | null>(null);
  const [analyzedStrategy, setAnalyzedStrategy] = useState<string | null>(null);

  // --- EFFETTO CARICAMENTO MODELLI ---
  useEffect(() => {
    fetch(`${BASE_API_URL}/models/available`)
      .then((res) => res.json())
      .then((data: AvailableModelsResponse) => {
        setAvailableModels(data);
        if (data["6class"]?.length > 0) {
          setSelectedModel6Class(data["6class"][0].filename);
        }
      })
      .catch(() => setApiError("Impossibile caricare la lista dei modelli"));
  }, []);

  // --- UTILS ---
  const getCacheKey = (strat: string, crop: string, gpu: boolean, occ: boolean, ig: boolean, modelName: string) => 
    `${strat}-${crop}-${gpu}-${occ}-${ig}-${modelName}`;

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

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    
    const currentKey = getCacheKey(modelStrategy, cropMode, useGpu, showOcclusion, showIG, selectedModel6Class);
    if (resultCache[currentKey]) {
      setResult(resultCache[currentKey]);
      setAnalyzedMode(cropMode);
      setAnalyzedStrategy(modelStrategy);
      return;
    }

    setLoading(true);
    setApiError(null);

    try {
      const endpoint = modelStrategy === "standard" ? "6class" : "1vsall";
      let finalData: ApiResponse;

      if (cropMode === "compare") {
        const bodyOriginal = new FormData();
        bodyOriginal.append("image", selectedFile);
        bodyOriginal.append("use_crop", "false");
        if (modelStrategy === "standard") bodyOriginal.append("model_name", selectedModel6Class);

        const bodyCropped = new FormData();
        bodyCropped.append("image", selectedFile);
        bodyCropped.append("use_crop", "true");
        if (modelStrategy === "standard") bodyCropped.append("model_name", selectedModel6Class);

        const [resOrig, resCrop] = await Promise.all([
          fetch(`${BASE_API_URL}/${endpoint}`, { method: "POST", body: bodyOriginal }),
          fetch(`${BASE_API_URL}/${endpoint}`, { method: "POST", body: bodyCropped })
        ]);

        if (!resOrig.ok || !resCrop.ok) throw new Error("Errore durante il confronto");

        const dataOrig: ApiResponse = await resOrig.json();
        const dataCrop: ApiResponse = await resCrop.json();

        finalData = {
          ...dataOrig,
          predicted_class_cropped: dataCrop.predicted_class,
          confidence_cropped: dataCrop.confidence,
          all_classes_probs_cropped: dataCrop.all_classes_probs,
          image_cropped: dataCrop.image_cropped,
        };
      } else {
        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("use_crop", cropMode === "external" ? "true" : "false");
        if (modelStrategy === "standard") formData.append("model_name", selectedModel6Class);

        const res = await fetch(`${BASE_API_URL}/${endpoint}`, { method: "POST", body: formData });
        if (!res.ok) throw new Error(`Errore Server: ${res.statusText}`);
        finalData = await res.json();
      }

      // --- EXPLAINABILITY ---
      if (useGpu && (showOcclusion || showIG)) {
        const explainFormData = new FormData();
        explainFormData.append("image", selectedFile);
        explainFormData.append("use_crop", cropMode === "external" ? "true" : "false");
        explainFormData.append("model_name", selectedModel6Class);

        if (showOcclusion) {
          const resOcc = await fetch(`${BASE_API_URL}/generate_occlusion`, { method: "POST", body: explainFormData });
          if (resOcc.ok) {
            const dataOcc = await resOcc.json();
            finalData.occlusion = dataOcc.explanation_image;
          }
        }
        if (showIG) {
          const resIG = await fetch(`${BASE_API_URL}/generate_explain`, { method: "POST", body: explainFormData });
          if (resIG.ok) {
            const dataIG = await resIG.json();
            finalData.integrated_gradients = dataIG.explanation_image;
          }
        }
      }

      setResult(finalData);
      setAnalyzedMode(cropMode);
      setAnalyzedStrategy(modelStrategy);
      setResultCache(prev => ({ ...prev, [currentKey]: finalData }));

    } catch (err) {
      // --- FIX ERRORE 154:19 ---
      // In TypeScript, gli errori catturati sono di tipo 'unknown'
      const errorMessage = err instanceof Error ? err.message : "Errore sconosciuto";
      setApiError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-screen bg-[#F6F4EF] text-[#2A2F2C] flex flex-col md:flex-row font-sans overflow-hidden">
      <DashboardSidebar
        config={{ 
          modelStrategy, 
          cropMode, 
          useGpu, 
          showOcclusion, 
          showIG, 
          selectedModel6Class 
        }}
        availableModels={availableModels}
        setConfig={{ 
          setModelStrategy, 
          setCropMode, 
          setUseGpu, 
          setShowOcclusion, 
          setShowIG, 
          setSelectedModel6Class 
        }}
        fileState={{ selectedFile, handleFileChange }}
        actionState={{ loading, handleAnalyze, apiError }}
      />

      <main className="flex-1 p-8 md:p-12 overflow-y-auto bg-pink-50 relative">
        <header className="mb-10">
          <h1 className="text-3xl md:text-4xl font-extrabold text-emerald-900 mb-2 flex items-center gap-3">
            <span className="text-pink-600">🌸</span> Inference for orchids
          </h1>
          <p className="text-stone-500 font-medium italic">Seleziona un modello e analizza la specie</p>
        </header>

        {preview && !result && (
           <div className="max-w-4xl mx-auto border-2 border-dashed border-stone-300 rounded-2xl p-8 flex flex-col items-center justify-center bg-stone-50/50">
             <div className="relative w-full h-[500px] mb-4 p-2 bg-white rounded-xl shadow-sm border border-stone-100">
               <Image src={preview} alt="Upload" fill className="object-contain rounded-lg" unoptimized />
             </div>
             {loading && <p className="animate-pulse text-emerald-700 font-bold text-lg">⚙️ Analisi in corso...</p>}
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
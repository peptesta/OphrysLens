import { ChangeEvent } from "react";

interface ModelInfo {
  id: number;
  filename: string;
  label: string;
}

interface DashboardSidebarProps {
  // Config State
  config: {
    modelStrategy: string;
    cropMode: string;
    useGpu: boolean;
    showOcclusion: boolean;
    showIG: boolean;
    selectedModel6Class?: string;
  };
  // Modelli disponibili caricati dal backend (ora solo 6class)
  availableModels: {
    "6class": ModelInfo[];
  };
  // Config Setters
  setConfig: {
    setModelStrategy: (v: string) => void;
    setCropMode: (v: string) => void;
    setUseGpu: (v: boolean) => void;
    setShowOcclusion: (v: boolean) => void;
    setShowIG: (v: boolean) => void;
    setSelectedModel6Class: (v: string) => void;
  };
  // File & Actions
  fileState: {
    selectedFile: File | null;
    handleFileChange: (e: ChangeEvent<HTMLInputElement>) => void;
  };
  actionState: {
    loading: boolean;
    handleAnalyze: () => void;
    apiError: string | null;
  };
}

export default function DashboardSidebar({
  config,
  availableModels,
  setConfig,
  fileState,
  actionState,
}: DashboardSidebarProps) {
  return (
    <aside className="w-full md:w-80 bg-[#F0F7F3] border-r border-[#D8D2C8] flex flex-col flex-shrink-0 h-[100%] shadow-md">
      <div className="flex-1 overflow-y-auto p-6 space-y-6 custom-scrollbar">
        <div>
          <h2 className="text-lg font-bold flex items-center gap-2 mb-6 text-emerald-900">
            ⚙️ Control Panel
          </h2>

          <div className="space-y-5">
            {/* Strategy Selection */}
            <div>
              <label className="block text-xs font-bold text-stone-600 uppercase mb-1">
                Model Strategy
              </label>
              <select
                value={config.modelStrategy}
                onChange={(e) => setConfig.setModelStrategy(e.target.value)}
                className="w-full p-2.5 border border-stone-300 rounded-lg bg-white text-sm text-stone-700 focus:ring-2 focus:ring-emerald-500 outline-none transition"
              >
                <option value="standard">Standard 6-Class Model</option>
                <option value="1vsall">1-vs-All Ensemble</option>
              </select>
            </div>

            {/* Dinamic Selection for 6-Class */}
            {config.modelStrategy === "standard" && (
              <div className="animate-in fade-in slide-in-from-top-2 duration-300">
                <label className="block text-xs font-bold text-emerald-700 uppercase mb-1">
                  Select 6-Class Model
                </label>
                <select
                  value={config.selectedModel6Class}
                  onChange={(e) => setConfig.setSelectedModel6Class(e.target.value)}
                  className="w-full p-2.5 border border-emerald-200 rounded-lg bg-white text-sm text-stone-700 focus:ring-2 focus:ring-emerald-500 outline-none transition"
                >
                  <option value="">-- Choose a version --</option>
                  {availableModels["6class"]?.map((model) => (
                    <option key={model.id} value={model.filename}>
                      {model.label}
                    </option>
                  ))}
                </select>
              </div>
            )}

            {/* Static View for 1vsall */}
            {config.modelStrategy === "1vsall" && (
              <div className="animate-in fade-in slide-in-from-top-2 duration-300">
                <label className="block text-xs font-bold text-emerald-700 uppercase mb-1">
                  Active Model
                </label>
                <div className="p-2.5 border border-emerald-100 rounded-lg bg-emerald-50/30 text-sm text-emerald-800 font-medium">
                  🛡️ Multi-Model Ensemble Active
                </div>
                <p className="mt-1 text-[10px] text-stone-500 italic">
                  This strategy uses all specialized detectors simultaneously.
                </p>
              </div>
            )}

            {/* Crop Mode */}
            <div>
              <label className="block text-xs font-bold text-stone-600 uppercase mb-1">
                Cropping Mode
              </label>
              <select
                value={config.cropMode}
                onChange={(e) => setConfig.setCropMode(e.target.value)}
                className="w-full p-2.5 border border-stone-300 rounded-lg bg-white text-sm text-stone-700 focus:ring-2 focus:ring-emerald-500 outline-none transition"
              >
                <option value="integrated">Integrated (Standard)</option>
                <option value="external">External (Smart Crop)</option>
                <option value="compare">Compare Both</option>
              </select>
            </div>
          </div>
        </div>

        <hr className="border-stone-200" />

        {/* AI Settings Section */}
        <div>
          <h3 className="text-xs font-bold text-stone-600 uppercase mb-3">AI Settings</h3>
          <label className="flex items-center gap-3 mb-3 cursor-pointer p-2 hover:bg-white/60 rounded transition">
            <input
              type="checkbox"
              checked={config.useGpu}
              onChange={(e) => setConfig.setUseGpu(e.target.checked)}
              className="w-4 h-4 accent-emerald-600 rounded cursor-pointer"
            />
            <span className="text-sm font-medium text-stone-700">Enable Explainability</span>
          </label>

          {config.useGpu && (
            <div className="ml-2 pl-4 border-l-2 border-stone-300 space-y-2">
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.showOcclusion}
                  onChange={(e) => setConfig.setShowOcclusion(e.target.checked)}
                  className="w-3 h-3 accent-emerald-500"
                />
                <span className="text-xs text-stone-600">Occlusion Heatmap</span>
              </label>
              <label className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={config.showIG}
                  onChange={(e) => setConfig.setShowIG(e.target.checked)}
                  className="w-3 h-3 accent-emerald-500"
                />
                <span className="text-xs text-stone-600">Integrated Gradients</span>
              </label>
            </div>
          )}
        </div>

        <hr className="border-stone-200" />

        {/* File Upload */}
        <div>
          <label className="block w-full cursor-pointer bg-white/80 border-2 border-dashed border-stone-300 hover:border-emerald-500 hover:bg-emerald-50/50 text-stone-600 font-medium py-4 px-4 rounded-xl text-center transition group">
            <span className="block text-2xl mb-1 group-hover:scale-110 transition-transform duration-300">📂</span>
            <span className="text-sm">Choose Photo...</span>
            <input type="file" onChange={fileState.handleFileChange} accept="image/*" className="hidden" />
          </label>
          {fileState.selectedFile && (
            <div className="mt-3 p-2 bg-stone-100 text-stone-700 text-xs rounded border border-stone-200 truncate font-medium">
              Selected: {fileState.selectedFile.name}
            </div>
          )}
        </div>

        {/* Analyze Action */}
        <div className="space-y-3 pt-2">
          <button
            onClick={actionState.handleAnalyze}
            disabled={!fileState.selectedFile || actionState.loading}
            className="w-full bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 px-4 rounded-xl shadow-lg shadow-emerald-200/50 transition-all transform active:scale-[0.98] disabled:opacity-50 disabled:bg-stone-400"
          >
            {actionState.loading ? (
              <span className="flex items-center justify-center gap-2">
                <div className="animate-spin h-5 w-5 border-2 border-white border-t-transparent rounded-full" />
                Analyzing...
              </span>
            ) : (
              "🔍 Identify Species"
            )}
          </button>

          {actionState.apiError && (
            <div className="bg-red-50 text-red-700 p-3 rounded-lg text-xs border border-red-200 font-medium text-center">
              ❌ {actionState.apiError}
            </div>
          )}
        </div>
      </div>
    </aside>
  );
}
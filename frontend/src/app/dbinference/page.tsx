"use client";

import { useState } from "react";
import { BackendResponse, ImageFile } from "@/types/index";

// Components
import UploadSection from "@/components/UploadSection";
import ResultsList from "@/components/ResultsList";
import ImageEditorModal from "@/components/ImageEditorModal";
import SaveOptionsDropdown from "@/components/Save/SaveOptionsDropdown";
import OrganizationModal from "@/components/Save/OrganizationModal";

// Utilities
import { SaveStrategy, getBoxesForStrategy } from "@/utils/saveStrategies";

// --- TYPES ---
interface SaveMetadata {
  filename: string;
  boxes: number[][]; 
}

const ITEMS_PER_PAGE = 10;
const BATCH_SIZE = 16; 
const API_URL = 'http://localhost:5000/dbinference';
const SAVE_API_URL = 'http://localhost:5000/save_dataset'; 

export default function FolderTransformer() {
  // --- STATE ---
  const [files, setFiles] = useState<ImageFile[]>([]);
  const [transformedFiles, setTransformedFiles] = useState<ImageFile[]>([]);
  const [originalTransformedFiles, setOriginalTransformedFiles] = useState<ImageFile[]>([]);

  const [isTransforming, setIsTransforming] = useState(false);
  const [processedCount, setProcessedCount] = useState(0);

  // Save UI States
  const [folderName, setFolderName] = useState<string>("processed_dataset");
  const [isSavingRemote, setIsSavingRemote] = useState(false);
  const [isSaveDropdownOpen, setIsSaveDropdownOpen] = useState(false);
  const [showOrganizeModal, setShowOrganizeModal] = useState(false);
  
  // Save Logic States
  const [pendingSaveStrategy, setPendingSaveStrategy] = useState<SaveStrategy | null>(null);
  const [pendingThreshold, setPendingThreshold] = useState<number | undefined>(undefined);

  // Results UI States
  const [showResults, setShowResults] = useState(false);
  const [selectedImage, setSelectedImage] = useState<ImageFile | null>(null);
  
  // Pagination
  const [currentPage, setCurrentPage] = useState(1);
  const indexOfLastItem = currentPage * ITEMS_PER_PAGE;
  const indexOfFirstItem = indexOfLastItem - ITEMS_PER_PAGE;
  const currentFiles = transformedFiles.slice(indexOfFirstItem, indexOfLastItem);
  const totalPages = Math.ceil(transformedFiles.length / ITEMS_PER_PAGE);

  // --- ACTIONS ---

  const handleBackToUpload = () => {
    if (window.confirm("Are you sure? All current results and edits will be lost.")) {
      setFiles([]);
      setTransformedFiles([]);
      setOriginalTransformedFiles([]);
      setProcessedCount(0);
      setCurrentPage(1);
      setIsTransforming(false);
      setIsSavingRemote(false);
      setIsSaveDropdownOpen(false);
      setShowOrganizeModal(false);
      setPendingSaveStrategy(null);
      setPendingThreshold(undefined);
      setShowResults(false);
    }
  };

  const handleSaveChanges = (updatedImage: ImageFile) => {
    setTransformedFiles(prev => prev.map(f => f.name === updatedImage.name ? updatedImage : f));
    setSelectedImage({ ...updatedImage });
  };

  const handleGlobalReset = () => {
    if (originalTransformedFiles.length === 0) return;
    if (window.confirm("Reset ALL boxes for ALL images? This cannot be undone.")) {
      setTransformedFiles(JSON.parse(JSON.stringify(originalTransformedFiles)));
      alert("All images reset.");
    }
  };

  // --- SAVE FLOW ---
  
  const handleSaveClick = (strategy: SaveStrategy, threshold?: number) => {
    setIsSaveDropdownOpen(false);
    setPendingSaveStrategy(strategy);
    setPendingThreshold(threshold);

    if (strategy === 'global_best' && typeof threshold === 'undefined') {
      executeSaveRequest(strategy, false, undefined);
    } else {
      setShowOrganizeModal(true);
    }
  };

  const handleOrganizationConfirm = (organize: boolean) => {
    if (pendingSaveStrategy) {
      executeSaveRequest(pendingSaveStrategy, organize, pendingThreshold);
    }
  };

  const executeSaveRequest = async (
    strategy: SaveStrategy, 
    folderOrganized: boolean, 
    threshold?: number,
    resize?: { width: number; height: number }
  ) => {
    if (transformedFiles.length === 0) return;
    
    setIsSavingRemote(true);
    setShowOrganizeModal(false);
    
    const formData = new FormData();
    formData.append('folder_organized', String(folderOrganized));
    if (resize) {
      formData.append('resize_w', resize.width.toString());
      formData.append('resize_h', resize.height.toString());
    }

    try {
      const fileData = await Promise.all(
        transformedFiles.map(async (file) => {
          const res = await fetch(file.url);
          const blob = await res.blob();
          return { blob, name: file.name, originalFile: file };
        })
      );

      const metadata: SaveMetadata[] = [];

      fileData.forEach(({ blob, name, originalFile }) => {
        formData.append('images_files', blob, name);
        
        let baseStrategy = strategy;
        if (typeof threshold === 'number') {
          if (strategy === 'global_best') baseStrategy = 'global_all';
          if (strategy === 'custom_best') baseStrategy = 'custom_all';
        }

        let boxesToSend = getBoxesForStrategy(originalFile, baseStrategy);

        if (typeof threshold === 'number' && originalFile.analysis) {
          boxesToSend = boxesToSend.filter((_, idx) => {
            const score = originalFile.analysis!.scores[idx];
            const isManual = originalFile.analysis!.isManual?.[idx];
            return isManual || (score * 100) >= threshold;
          });
        }

        if (boxesToSend.length > 0) {
          metadata.push({ filename: name, boxes: boxesToSend });
        }
      });

      if (metadata.length === 0) {
        alert("Nessun crop corrisponde ai criteri di selezione.");
        setIsSavingRemote(false);
        return;
      }

      formData.append('metadata', JSON.stringify(metadata));
      
      const response = await fetch(SAVE_API_URL, { method: 'POST', body: formData });
      if (!response.ok) throw new Error("Errore nel salvataggio lato server");

      const zipBlob = await response.blob();
      const downloadUrl = window.URL.createObjectURL(zipBlob);
      const link = document.createElement('a');
      link.href = downloadUrl;
      
      const resizeLabel = resize ? `_${resize.width}x${resize.height}` : "";
      link.setAttribute('download', `${folderName}${resizeLabel}.zip`);
      
      document.body.appendChild(link);
      link.click();
      link.remove();
      window.URL.revokeObjectURL(downloadUrl);

    } catch (error) {
      console.error(error);
      alert("Errore durante la generazione del dataset.");
    } finally {
      setIsSavingRemote(false);
    }
  };

  // --- API: TRANSFORM ---
  const handleTransform = async () => {
    if (files.length === 0) return;
    setIsTransforming(true);
    setProcessedCount(0);
    
    let accumulatedCount = 0;
    const allTransformedResults: ImageFile[] = [];

    try {
      for (let i = 0; i < files.length; i += BATCH_SIZE) {
        const currentBatch = files.slice(i, i + BATCH_SIZE);
        const formData = new FormData();
        currentBatch.forEach(f => f.file && formData.append('images', f.file));

        const res = await fetch(API_URL, { method: 'POST', body: formData });
        if (!res.ok) throw new Error(res.statusText);
        const data: BackendResponse = await res.json();

        const batchResults: ImageFile[] = currentBatch.map((orig, idx) => ({
          name: orig.name,
          url: data.images[idx] ? (data.images[idx].startsWith('data:') ? data.images[idx] : `data:image/jpeg;base64,${data.images[idx]}`) : orig.url,
          analysis: {
            boxes: data.bounding_box?.[idx] || [],
            scores: data.scores?.[idx] || [],
            count: (data.bounding_box?.[idx] || []).length,
            modified: new Array((data.bounding_box?.[idx] || []).length).fill(false),
            isManual: new Array((data.bounding_box?.[idx] || []).length).fill(false),
            eliminated: new Array((data.bounding_box?.[idx] || []).length).fill(false)
          }
        }));

        allTransformedResults.push(...batchResults);
        accumulatedCount += currentBatch.length;
        setProcessedCount(accumulatedCount);
      }

      setTransformedFiles(allTransformedResults);
      setOriginalTransformedFiles(JSON.parse(JSON.stringify(allTransformedResults)));
      setShowResults(true);
      setCurrentPage(1);

    } catch (error) {
      console.error("Dettaglio Errore:", error);
      alert(`Errore: ${error instanceof Error ? error.message : 'Unknown error'}`);
    } finally {
      setIsTransforming(false);
    }
  };

  // --- UPLOAD ---
  const handleFolderUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (!e.target.files?.length) return;
    
    const allFiles = Array.from(e.target.files);
    
    if (allFiles[0].webkitRelativePath) {
      const pathParts = allFiles[0].webkitRelativePath.split('/');
      if (pathParts.length > 1) {
        setFolderName(pathParts[0]);
      }
    }

    const validImages = allFiles.filter(f => 
      f.type.startsWith("image/") && /\.(jpg|jpeg|png|webp)$/i.test(f.name)
    );
    
    if (validImages.length > 0) {
      setFiles(validImages.map(f => ({ 
        name: f.name, 
        url: URL.createObjectURL(f), 
        file: f 
      })));
      setShowResults(false);
      setProcessedCount(0);
    } else {
      alert("No valid images found.");
    }
  };

  return (
    <div className="min-h-screen bg-[#F6F4EF] text-[#2A2F2C] font-sans p-6 md:p-12 relative">
      <header className="max-w-5xl mx-auto mb-10 flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-extrabold text-emerald-900 flex items-center gap-3">
            <span className="text-pink-600">🌸</span> Image Batch Processor
          </h1>
          <p className="text-stone-500 font-medium mt-1">
            {showResults ? "Processing Results" : "Upload a folder to start"}
          </p>
        </div>
        
        {showResults && (
           <button 
             onClick={handleBackToUpload} 
             className="text-stone-500 hover:text-emerald-700 font-semibold text-sm underline transition"
           >
             ← Upload new files
           </button>
        )}
      </header>

      <main className="max-w-5xl mx-auto">
        {!showResults ? (
          <UploadSection 
            fileCount={files.length}
            isTransforming={isTransforming}
            processedCount={processedCount} 
            onFilesSelected={handleFolderUpload}
            onTransform={handleTransform}
          />
        ) : (
          <>
            <ResultsList 
              files={currentFiles}
              totalCount={transformedFiles.length}
              currentPage={currentPage}
              totalPages={totalPages}
              startIndex={indexOfFirstItem}
              endIndex={indexOfLastItem}
              itemsPerPage={ITEMS_PER_PAGE}
              onSelect={setSelectedImage}
              onPageChange={(dir) => {
                if (dir === 'next' && currentPage < totalPages) setCurrentPage(p => p + 1);
                if (dir === 'prev' && currentPage > 1) setCurrentPage(p => p - 1);
              }}
              onResetAll={handleGlobalReset}
            />

            <div className="flex justify-end mt-4">
              <SaveOptionsDropdown 
                isSaving={isSavingRemote}
                isOpen={isSaveDropdownOpen}
                onToggle={() => setIsSaveDropdownOpen(!isSaveDropdownOpen)}
                onClose={() => setIsSaveDropdownOpen(false)}
                onSelect={handleSaveClick}
              />
            </div>
          </>
        )}
      </main>

      <OrganizationModal 
        isOpen={showOrganizeModal}
        onClose={() => setShowOrganizeModal(false)}
        onConfirm={handleOrganizationConfirm}
      />

      {selectedImage && (
        <ImageEditorModal          
          selectedImage={selectedImage}
          onClose={() => setSelectedImage(null)}
          onSave={handleSaveChanges}
        />
      )}
    </div>
  );
}
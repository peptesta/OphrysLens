// --- TYPES ---
export interface ApiResponse {
  success: boolean;
  predicted_class: string;
  confidence: number;
  all_classes_probs: number[];
  image?: string; 
  image_cropped?: string; 
  
  // --- NUOVI CAMPI COMBINATI ---
  explanation_combined?: string | null; 
  explanation_combined_cropped?: string | null;

  // Vecchi campi (opzionali se vuoi migrare gradualmente)
  integrated_gradients?: string | null; 
  occlusion?: string | null; 

  predicted_class_cropped?: string;
  confidence_cropped?: number;
  all_classes_probs_cropped?: number[];
  integrated_gradients_cropped?: string | null; 
  occlusion_cropped?: string | null; 

  error?: string;
  traceback?: string;
}

export interface ImageFile {
  name: string;
  url: string;      
  file?: File;      
  analysis?: {
    boxes: number[][];    // Matrix Nx4
    scores: number[];     // List of N scores
    labels?: string[];    // Optional labels if your backend sends them
    count: number;        // Number N of detections
    
    // Tracking state
    modified?: boolean[];   // Tracks if a box position was edited
    eliminated?: boolean[]; // Tracks if a box was deleted/hidden
    isManual?: boolean[];   // Tracks if a box was manually drawn by the user
  };
}

export interface BackendResponse {
  images: string[];   
  bounding_box: number[][][]; 
  scores: number[][];         
  bb_count: number;           
}
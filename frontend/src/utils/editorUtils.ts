// src/utils/editorUtils.ts

// --- TYPES ---
export type ConflictType = 'overlap' | 'containment';

export interface Conflict {
    type: ConflictType;
    indices: number[]; // [indexA, indexB]
    message: string;
}

// --- HELPER FUNCTIONS ---

export const getScoreColor = (score: number, isManual: boolean) => {
  if (isManual) return "#3b82f6"; // Blue for manual
  if (score < 0.5) return "#ef4444"; 
  if (score < 0.7) return "#f97316"; 
  if (score < 0.85) return "#eab308"; 
  if (score < 0.95) return "#22c55e"; 
  return "#15803d"; 
};

const getArea = (box: number[]) => (box[2] - box[0]) * (box[3] - box[1]);

const getIntersectionArea = (boxA: number[], boxB: number[]) => {
    const x_overlap = Math.max(0, Math.min(boxA[2], boxB[2]) - Math.max(boxA[0], boxB[0]));
    const y_overlap = Math.max(0, Math.min(boxA[3], boxB[3]) - Math.max(boxA[1], boxB[1]));
    return x_overlap * y_overlap;
};

// Conflict detection logic
export const detectConflicts = (boxes: number[][], eliminated: boolean[]): Conflict[] => {
    const conflicts: Conflict[] = [];
    for (let i = 0; i < boxes.length; i++) {
        if (eliminated[i]) continue; 

        for (let j = i + 1; j < boxes.length; j++) {
            if (eliminated[j]) continue; 

            const boxA = boxes[i];
            const boxB = boxes[j];
            const areaA = getArea(boxA);
            const areaB = getArea(boxB);
            const intersection = getIntersectionArea(boxA, boxB);

            if (intersection > 0) {
                const isAInsideB = boxA[0] >= boxB[0] && boxA[1] >= boxB[1] && boxA[2] <= boxB[2] && boxA[3] <= boxB[3];
                const isBInsideA = boxB[0] >= boxA[0] && boxB[1] >= boxA[1] && boxB[2] <= boxA[2] && boxB[3] <= boxA[3];

                if (isAInsideB || isBInsideA) {
                    conflicts.push({ type: 'containment', indices: [i, j], message: `Box #${i+1} is inside Box #${j+1}` });
                } else {
                    const smallerArea = Math.min(areaA, areaB);
                    const overlapRatio = intersection / smallerArea;
                    if (overlapRatio > 0.5) {
                        conflicts.push({ type: 'overlap', indices: [i, j], message: `High Overlap (${(overlapRatio * 100).toFixed(0)}%) between #${i+1} & #${j+1}` });
                    }
                }
            }
        }
    }
    return conflicts;
};
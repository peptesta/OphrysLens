import { ImageFile } from "@/types/index";

export type SaveStrategy = 'global_best' | 'global_all' | 'custom_best' | 'custom_all';

/**
 * Checks if a file has user modifications (manual boxes, eliminations, or edits).
 */
const isFileModified = (file: ImageFile): boolean => {
  const a = file.analysis;
  if (!a) return false;
  return (
    (a.modified?.some(b => b) ?? false) ||
    (a.eliminated?.some(b => b) ?? false) ||
    (a.isManual?.some(b => b) ?? false)
  );
};

/**
 * Determines which bounding boxes to return based on the selected strategy.
 */
export const getBoxesForStrategy = (file: ImageFile, strategy: SaveStrategy): number[][] => {
  if (!file.analysis) return [];

  // 1. Get all currently visible boxes (filtering out eliminated ones)
  const activeIndices = file.analysis.boxes.map((_, i) => i).filter(i => !file.analysis?.eliminated?.[i]);
  const scores = file.analysis.scores || [];

  const getBestBox = () => {
    if (activeIndices.length === 0) return [];
    let bestIdx = -1;
    let maxScore = -1;
    activeIndices.forEach(idx => {
      if (scores[idx] > maxScore) {
        maxScore = scores[idx];
        bestIdx = idx;
      }
    });
    return bestIdx !== -1 ? [file.analysis!.boxes[bestIdx]] : [];
  };

  const getAllActiveBoxes = () => {
    return activeIndices.map(idx => file.analysis!.boxes[idx]);
  };

  switch (strategy) {
    case 'global_best':
      return getBestBox();

    case 'global_all':
      return getAllActiveBoxes();

    case 'custom_best':
      // Modified? -> Save User's Work. Unmodified? -> Save Best.
      return isFileModified(file) ? getAllActiveBoxes() : getBestBox();

    case 'custom_all':
      // Modified? -> Save User's Work. Unmodified? -> Save All Original.
      return getAllActiveBoxes();

    default:
      return getAllActiveBoxes();
  }
};
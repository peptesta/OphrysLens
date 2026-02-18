import { X, FolderInput, FileStack } from "lucide-react";
import { ReactNode } from "react";

interface OrganizationModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: (organizeIntoFolders: boolean) => void;
}

// 1. Definiamo l'interfaccia per il sotto-componente
interface SelectionButtonProps {
  onClick: () => void;
  icon: ReactNode;
  title: string;
  subtitle: string;
  theme: 'emerald' | 'stone';
}

export default function OrganizationModal({ isOpen, onClose, onConfirm }: OrganizationModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/40 backdrop-blur-sm z-50 flex items-center justify-center p-4">
      <div className="bg-white rounded-2xl shadow-2xl max-w-md w-full overflow-hidden animate-in fade-in zoom-in-95 duration-200">
        <div className="p-6">
          <div className="flex justify-between items-start mb-4">
            <h3 className="text-xl font-bold text-emerald-900">Organize Output?</h3>
            <button onClick={onClose} className="text-stone-400 hover:text-stone-600 transition">
              <X size={24} />
            </button>
          </div>
          
          <p className="text-stone-600 mb-8 leading-relaxed">
            You are about to save multiple crops per image. Would you like to organize the files into separate folders for each original image?
          </p>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <SelectionButton 
              onClick={() => onConfirm(true)}
              icon={<FolderInput size={24} />}
              title="Yes"
              subtitle="Create Folders"
              theme="emerald"
            />
            <SelectionButton 
              onClick={() => onConfirm(false)}
              icon={<FileStack size={24} />}
              title="No"
              subtitle="Flat List"
              theme="stone"
            />
          </div>
        </div>
      </div>
    </div>
  );
}

// 2. Applichiamo l'interfaccia al sub-componente rimuovendo 'any'
const SelectionButton = ({ onClick, icon, title, subtitle, theme }: SelectionButtonProps) => {
  const isEmerald = theme === 'emerald';
  return (
    <button
      onClick={onClick}
      className={`flex items-center justify-center gap-3 border p-4 rounded-xl transition group text-left w-full
        ${isEmerald 
          ? 'bg-emerald-50 border-emerald-200 hover:bg-emerald-100 hover:border-emerald-300' 
          : 'bg-stone-50 border-stone-200 hover:bg-stone-100 hover:border-stone-300'}`}
    >
      <div className={`p-2 rounded-full transition ${isEmerald ? 'bg-emerald-200 text-emerald-800 group-hover:bg-emerald-300' : 'bg-stone-200 text-stone-600 group-hover:bg-stone-300'}`}>
        {icon}
      </div>
      <div>
        <span className={`block font-bold ${isEmerald ? 'text-emerald-900' : 'text-stone-800'}`}>{title}</span>
        <span className={`text-xs ${isEmerald ? 'text-emerald-700' : 'text-stone-500'}`}>{subtitle}</span>
      </div>
    </button>
  );
};
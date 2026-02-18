import { useState, useEffect } from 'react';
import exifr from 'exifr';

// 1. Definiamo un'interfaccia per i metadati comuni
export interface ImageMetadata {
  latitude?: number;
  longitude?: number;
  DateTimeOriginal?: Date | string;
  Make?: string;
  Model?: string;
  Software?: string;
  ImageWidth?: number;
  ImageHeight?: number;
  // Permettiamo altri campi dinamici senza usare 'any'
  [key: string]: string | number | Date | boolean | undefined;
}

export function useImageMetadata(imageSrc: string | null) {
  // 2. Applichiamo l'interfaccia allo stato
  const [metadata, setMetadata] = useState<ImageMetadata | null>(null);
  const [address, setAddress] = useState<string | null>(null);
  const [loading, setLoading] = useState<boolean>(false);

  useEffect(() => {
    async function extract() {
      if (!imageSrc) {
        setMetadata(null);
        setAddress(null);
        return;
      }
      
      setLoading(true);
      setAddress(null);

      try {
        const data = await exifr.parse(imageSrc, {
          tiff: true, 
          xmp: true, 
          gps: true, 
          translateKeys: true,
        });

        if (data) {
          setMetadata(data as ImageMetadata);

          // Reverse Geocoding se esistono le coordinate
          if (data.latitude && data.longitude) {
            const res = await fetch(
              `https://nominatim.openstreetmap.org/reverse?format=json&lat=${data.latitude}&lon=${data.longitude}&zoom=10&addressdetails=1`,
              { headers: { 'Accept-Language': 'it' } }
            );
            const geoData = await res.json();
            
            // Estrazione sicura dei dati geografici
            const addr = geoData.address;
            const city = addr.city || addr.town || addr.village || addr.suburb;
            const country = addr.country;
            setAddress(city ? `${city}, ${country}` : country);
          }
        }
      } catch (e: unknown) {
        // Gestione errore tipizzata
        const errorMsg = e instanceof Error ? e.message : "Unknown EXIF error";
        console.warn("Errore EXIF:", errorMsg);
        setMetadata(null);
      } finally {
        setLoading(false);
      }
    }
    extract();
  }, [imageSrc]);

  return { metadata, address, loading };
}
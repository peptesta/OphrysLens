"use client";

import Image from "next/image";
import { useState, useEffect } from "react";

// List of images to display as animated thumbnails
const THUMBNAILS = [
  "images/orch1.jpg",
  "images/orch2.jpg",
  "images/orch3.jpg",
  "images/orch4.jpg",
  "images/orch5.jpg",
  "images/orch6.jpg",
  "images/orch7.jpg",
  "images/orch8.jpg",
  "images/orch9.jpg",
];

type FloatingImage = {
  src: string;
  size: number;
  top: number;
  left: number;
  velocityX: number;
  velocityY: number;
};

export default function LandingPage() {
  const [floatingImages, setFloatingImages] = useState<FloatingImage[]>([]);

  // Initialize images with random size, position, and velocity
  useEffect(() => {
    const containerWidth = 800; // You can replace this with a dynamic ref if desired
    const containerHeight = 500;

    const imgs: FloatingImage[] = THUMBNAILS.map((src) => {
      const size = 150 + Math.random() * 150; // Size between 150-300px
      return {
        src,
        size,
        top: Math.random() * (containerHeight - size),
        left: Math.random() * (containerWidth - size),
        velocityX: (Math.random() - 0.5) * 2, // Velocity between -1 and 1 px per frame
        velocityY: (Math.random() - 0.5) * 2,
      };
    });

    setFloatingImages(imgs);
  }, []);

  // Bouncing animation logic
  useEffect(() => {
    const container = document.getElementById("floating-container");
    if (!container) return;

    let animationFrame: number;

    const animate = () => {
      setFloatingImages(prev =>
        prev.map(img => {
          let newLeft = img.left + img.velocityX;
          let newTop = img.top + img.velocityY;

          // Bounce off the edges
          if (newLeft <= 0 || newLeft + img.size >= container.offsetWidth) {
            img.velocityX *= -1;
            newLeft = Math.max(0, Math.min(newLeft, container.offsetWidth - img.size));
          }
          if (newTop <= 0 || newTop + img.size >= container.offsetHeight) {
            img.velocityY *= -1;
            newTop = Math.max(0, Math.min(newTop, container.offsetHeight - img.size));
          }

          return { ...img, left: newLeft, top: newTop };
        })
      );

      animationFrame = requestAnimationFrame(animate);
    };

    animationFrame = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(animationFrame);
  }, []);

  return (
    <div className="min-h-screen bg-[#F6F4EF] text-[#2A2F2C] font-sans flex flex-col items-center justify-start">
      
      {/* Header */}
      <header className="w-full bg-[#F0F7F3] border-b border-[#D8D2C8] shadow-md p-8 flex flex-col items-center">
        <h1 className="text-5xl md:text-6xl font-extrabold text-emerald-900 flex items-center gap-3">
          <span className="text-pink-600">🌿</span> Welcome to OphrysLens
        </h1>
      </header>

      {/* Hero Section */}
      <section className="flex-1 w-full flex flex-col items-center justify-center p-12">
        <div className="max-w-4xl w-full text-center space-y-6">
          <h2 className="text-3xl md:text-4xl font-bold text-emerald-800">
            Explore the Beauty of Orchids with AI
          </h2>
          <p className="text-stone-500 text-base md:text-lg">
            Even though the following orchids look alike, we are actually looking at 5 different species! 
          </p>
          <p className="text-stone-500 text-base md:text-lg">
            Our AI helps us distinguish them just by looking at the image          
          </p>
          
          {/* Container with animated thumbnails */}
          <div
            id="floating-container"
            className="relative w-full h-[500px] md:h-[600px] bg-stone-100 rounded-2xl overflow-hidden border border-stone-200 group"
          >
            {/* --- UPDATED LOGIC HERE --- */}
            {floatingImages.length === 0 ? (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-stone-400 select-none">
                <p className="text-2xl font-bold mb-2">Reload the page to see the magic ✨</p>
                <button 
                  onClick={() => window.location.reload()} 
                  className="text-sm underline hover:text-emerald-600 transition-colors"
                >
                  (Click here to reload)
                </button>
              </div>
            ) : (
              floatingImages.map((img, idx) => (
                <div
                  key={idx}
                  className="absolute rounded-xl overflow-hidden shadow-lg"
                  style={{
                    top: `${img.top}px`,
                    left: `${img.left}px`,
                    width: `${img.size}px`,
                    height: `${img.size}px`,
                  }}
                >
                  <Image
                    src={img.src}
                    alt={`Thumbnail ${idx}`}
                    fill
                    className="object-cover"
                    unoptimized
                  />
                </div>
              ))
            )}
            {/* -------------------------- */}
          </div>

          {/* Call to Action */}
          <button
            className="mt-6 bg-emerald-600 hover:bg-emerald-700 text-white font-bold py-3 px-6 rounded-xl shadow-lg shadow-emerald-200/50 transition-all transform active:scale-[0.98]"
            onClick={() => {
              window.location.href = '/inference';
            }}
          >
            Get Started
          </button>
        </div>
      </section>
    </div>
  );
}
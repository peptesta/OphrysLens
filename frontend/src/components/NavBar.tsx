'use client';

import Image from 'next/image';
import Link from 'next/link';
import { usePathname } from 'next/navigation';
import { useState } from 'react';

export default function NavBar() {
  const [isOpen, setIsOpen] = useState(false);

  const links = [
    { href: '/', label: 'Home' },
    { href: '/inference', label: 'Inference' },
    { href: '/dbinference', label: 'Database Transfomation' },
  ];

  const pathname = usePathname();

  return (
    <div className="px-4 md:px-8 bg-[#f4e8d5] flex items-center justify-between h-[70px] relative shadow-sm z-50">
      
      {/* Logo */}
      <div className="flex items-center shrink-0">
        <Image
          src="/logo.png"
          alt="OphrysLens Logo"
          width={40}
          height={40}
        />
        <h1 className="ml-3 text-xl md:text-2xl font-extrabold text-emerald-900 flex items-center gap-2">
            OrchiTech
        </h1>  
      </div>

      {/* Navigation Links */}
      <div className="hidden md:flex items-center gap-8 absolute left-1/2 transform -translate-x-1/2 h-full">
        {links.map((link) => {
          const isActive = pathname === link.href;
          return (
            <Link
              href={link.href}
              key={link.label}
              className={
                isActive
                  ? "text-emerald-800 font-bold border-b-2 border-emerald-600 px-2 h-full flex items-center transition-colors"
                  : "text-stone-600 hover:text-emerald-700 font-medium px-2 h-full flex items-center transition-colors hover:bg-stone-300/50"
              }
            >
              {link.label}
            </Link>
          );
        })}
      </div> 

      <div className="hidden md:block w-[40px]"></div>

      {/* Mobile menu button */}
      <div className="md:hidden ml-auto">
        <button
          onClick={() => setIsOpen(!isOpen)}
          className="p-2 rounded-md text-stone-700 hover:bg-stone-300 focus:outline-none transition"
        >
          {isOpen ? (
            <svg className="w-8 h-8" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
            </svg>
          ) : (
            <svg className="w-8 h-8" fill="none" stroke="currentColor" strokeWidth={2} viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          )}
        </button>
      </div>

      {/* Mobile menu (Dropdown) */}
      {isOpen && (
        <div className="absolute top-[70px] left-0 w-full bg-stone-100 shadow-xl flex flex-col items-start md:hidden border-t border-stone-300">
          {links.map((link) => (
            <Link
              href={link.href}
              key={link.label}
              className={
                pathname === link.href
                  ? "w-full px-6 py-4 text-lg font-bold text-emerald-900 bg-stone-200 border-l-4 border-emerald-600"
                  : "w-full px-6 py-4 text-lg font-medium text-stone-600 hover:bg-stone-200 hover:text-emerald-800 transition"
              }
              onClick={() => setIsOpen(false)}
            >
              {link.label}
            </Link>
          ))}
        </div>
      )}
    </div>
  );
}

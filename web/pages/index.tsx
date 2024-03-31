import React from 'react';
import { SparklesCore } from '../components/Background/SparklesCore';
import UploadInterface from '../components/Upload/UploadInterface';

export default function Home() {
  return (
    <div className="flex flex-col min-h-screen">
      <div className="flex-grow relative">
        <SparklesCore
          background="#0d47a1"
          minSize={1}
          maxSize={2}
          speed={4}
          particleColor="#000000"
          particleDensity={200}
          className="h-full"
        />
        <div className="absolute inset-0 flex items-center justify-center">
          <UploadInterface />
        </div>
      </div>
    </div>
  );
}
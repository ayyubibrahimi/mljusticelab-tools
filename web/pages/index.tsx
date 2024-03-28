import React, { useState, useEffect, useRef } from 'react';
import ChatInterface from '../components/ChatInterface';


export default function Home() {
  const [processingStatus, setProcessingStatus] = useState<'idle' | 'processing' | 'completed'>('idle');
  const [processingOutput, setProcessingOutput] = useState<string>('');
  const outputRef = useRef<HTMLDivElement>(null);

  const handleFileUpload = async (file: File) => {
    setProcessingStatus('processing');
    setProcessingOutput('');

    const formData = new FormData();
    formData.append('file', file);

    try {
      const response = await fetch('/api/upload', {
        method: 'POST',
        body: formData,
      });

      if (response.ok) {
        const reader = response.body?.getReader();
        const decoder = new TextDecoder('utf-8');

        while (true) {
          const { value, done } = await reader?.read() || {};
          if (done) break;

          const chunk = decoder.decode(value);
          setProcessingOutput((prevOutput) => prevOutput + chunk);
        }

        setProcessingStatus('completed');
      } else {
        setProcessingStatus('idle');
        console.error('File processing failed');
      }
    } catch (error) {
      console.error(error);
      setProcessingStatus('idle');
    }
  };

  useEffect(() => {
    if (outputRef.current) {
      outputRef.current.scrollTop = outputRef.current.scrollHeight;
    }
  }, [processingOutput]);

  return (
    <ChatInterface
      onFileUpload={handleFileUpload}
      processingStatus={processingStatus}
      processingOutput={processingOutput}
    />
  );
}
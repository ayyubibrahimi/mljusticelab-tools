import { NextApiRequest, NextApiResponse } from 'next';
import multer from 'multer';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

const upload = multer({ dest: 'public/uploads/' });

export const config = {
  api: {
    bodyParser: false,
  },
};

const scriptsDir = path.join(process.cwd(), 'pages/api');

export default async function handler(req: NextApiRequest, res: NextApiResponse) {
  if (req.method === 'POST') {
    upload.single('file')(req, res, async (err) => {
      if (err) {
        res.status(500).json({ error: 'File upload failed' });
        return;
      }

      const file = req.file;
      if (!file) {
        res.status(400).json({ error: 'No file uploaded' });
        return;
      }

      const tempFilePath = `public/uploads/${file.filename}.pdf`;
      fs.renameSync(file.path, tempFilePath);
      const tempOutputPath = `public/uploads/${file.filename}.json`;

      res.writeHead(200, {
        'Content-Type': 'application/json',
        'Transfer-Encoding': 'chunked',
      });

      // Process the PDF with ocr.py script
      const ocrScriptPath = path.join(scriptsDir, 'ocr.py');
      const ocrProcess = spawn('python3', [ocrScriptPath, tempFilePath, tempOutputPath]);

      ocrProcess.stderr.on('data', (data) => {
        console.error(`OCR script error: ${data}`);
      });

      ocrProcess.on('close', (code) => {
        
        if (code === 0) {
          const selectedScript = req.body.script; // Get the selected script from the request body
          const selectedModel = req.body.model; // Get the selected model from the request body

          if (selectedScript === 'process.py') {
            // Pass the output file path of ocr.py to process.py script
            const processScriptPath = path.join(scriptsDir, 'process.py');
            const myenvPath = path.join(scriptsDir, 'myenv', 'bin', 'python3');
            const processProcess = spawn(myenvPath, [processScriptPath, tempOutputPath, selectedModel]);

            let processOutput = '';

            processProcess.stdout.on('data', (data) => {
              processOutput += data.toString();
            });
            
            processProcess.stderr.on('data', (data) => {
              console.error('Process script error:', data.toString());
            });
            
            processProcess.on('close', (code) => {
              console.log('Process script output:', processOutput);
            
              if (code === 0) {
                try {
                  const sentencePagePairs = JSON.parse(processOutput);
                  if (sentencePagePairs !== null) {
                    console.log('Parsed sentence-page pairs:', sentencePagePairs);
                    const pdfFilePath = `uploads/${file.filename}.pdf`;
                    const responseData = JSON.stringify({ sentencePagePairs, filePath: pdfFilePath });
                    res.write(responseData);
                  } else {
                    console.error('Parsed sentence-page pairs is null');
                    const errorData = JSON.stringify({ error: 'Parsed sentence-page pairs is null' });
                    res.write(errorData);
                  }
                } catch (error) {
                  console.error('Error parsing process output:', error);
                  const errorData = JSON.stringify({ error: 'Error parsing process output', errorDetails: error.message });
                  res.write(errorData);
                }
                res.end();
              } else {
                console.error('Process script failed with code:', code);
                const errorData = JSON.stringify({ error: 'Process script failed', errorDetails: processOutput });
                res.write(errorData);
                res.end();
              }
            });
          } else if (selectedScript === 'toc.py') {
            // Pass the output file path of ocr.py to toc.py script
            const tocScriptPath = path.join(scriptsDir, 'toc.py');
            const myenvPath = path.join(scriptsDir, 'myenv', 'bin', 'python3');
            const tocProcess = spawn(myenvPath, [tocScriptPath, tempOutputPath]);
    
            let tocOutput = '';
            tocProcess.stdout.on('data', (data) => {
              tocOutput += data.toString();
            });

            tocProcess.stderr.on('data', (data) => {
              console.error(`TOC script error: ${data}`);
            });

            tocProcess.on('close', (code) => {
              if (code === 0) {
                try {
                  console.log('TOC output:', tocOutput);
                  const tocData = JSON.parse(tocOutput);
                  console.log('Parsed TOC data:', tocData);

                  if (Array.isArray(tocData)) {
                    const pdfFilePath = `public/uploads/${file.filename}.pdf`;
                    res.write(JSON.stringify({ tocData, filePath: pdfFilePath }));
                  } else {
                    console.error('Invalid TOC data format');
                    res.write(JSON.stringify({ error: 'Invalid TOC data format' }));
                  }
                } catch (error) {
                  console.error('Error parsing TOC output:', error);
                  res.write(JSON.stringify({ error: 'Error parsing TOC output' }));
                }
              } else {
                console.error('TOC script failed with code:', code);
                res.write(JSON.stringify({ error: 'TOC script failed' }));
              }
              res.end();
            });
          } else if (selectedScript === 'entity.py') {
            // Pass the output file path of ocr.py to entity.py script
            const entityScriptPath = path.join(scriptsDir, 'entity.py');
            const csvOutputPath = `public/uploads/${file.filename}.csv`;
            const entityProcess = spawn('python3', [entityScriptPath, tempOutputPath, csvOutputPath]);

            entityProcess.stderr.on('data', (data) => {
              console.error(`Entity script error: ${data}`);
            });

            entityProcess.on('close', (code) => {
              if (code === 0) {
                const csvFilePath = `/uploads/${file.filename}.csv`;
                res.write(JSON.stringify({ csvFilePath }));
              } else {
                console.error('Entity script failed with code:', code);
                res.write(JSON.stringify({ error: 'Entity script failed' }));
              }
              res.end();
            });
          } else {
            res.write(JSON.stringify({ error: 'Invalid script selected' }));
            res.end();
          }
        } else {
          console.error('OCR script failed with code:', code);
          res.end(JSON.stringify({ error: 'OCR script failed' }));
        }
      });
    });
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}
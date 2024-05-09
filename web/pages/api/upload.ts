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
    const uploadMiddleware = upload.fields([{ name: 'files', maxCount: 10 }]);

    uploadMiddleware(req, res, async (err) => {
      if (err) {
        res.status(500).json({ error: 'File upload failed' });
        return;
      }

      const files = req.files?.['files'] as Express.Multer.File[];
      if (!files || files.length === 0) {
        res.status(400).json({ error: 'No files uploaded' });
        return;
      }

      const selectedScript = req.body.script;
      const selectedModel = req.body.model;

      const ocrOutputDirectory = 'public/ocr_output/';
      if (!fs.existsSync(ocrOutputDirectory)) {
        fs.mkdirSync(ocrOutputDirectory);
      }

      // Create a new directory for the current batch of files
      const batchDirectory = `${ocrOutputDirectory}batch_${Date.now()}/`;
      fs.mkdirSync(batchDirectory);

      const ocrPromises = files.map((file) => {
        const tempFilePath = `public/uploads/${file.filename}.pdf`;
        fs.renameSync(file.path, tempFilePath);
        const tempOutputPath = `${batchDirectory}${file.originalname}.json`;

        const ocrScriptPath = path.join(scriptsDir, 'ocr.py');
        const ocrProcess = spawn('python3', [ocrScriptPath, tempFilePath, tempOutputPath]);

        return new Promise((resolve, reject) => {
          ocrProcess.stderr.on('data', (data) => {
            console.error(`OCR script error: ${data}`);
          });

          ocrProcess.on('close', (code) => {
            if (code === 0) {
              resolve(tempOutputPath);
            } else {
              reject(new Error('OCR script failed'));
            }
          });
        });
      });

      try {
        await Promise.all(ocrPromises);

        if (selectedScript === 'bulk_summary.py') {
          const bulkSummaryScriptPath = path.join(scriptsDir, 'bulk_summary.py');
          const myenvPath = path.join(scriptsDir, 'myenv', 'bin', 'python3');
          const bulkSummaryProcess = spawn(myenvPath, [bulkSummaryScriptPath, batchDirectory, selectedModel]);
          let bulkSummaryOutput = '';
        
          bulkSummaryProcess.stdout.on('data', (data) => {
            bulkSummaryOutput += data.toString();
            console.log('Bulk summary script output:', data.toString());
          });
        
          bulkSummaryProcess.stderr.on('data', (data) => {
            console.error('Bulk summary script error:', data.toString());
          });
        
          bulkSummaryProcess.on('close', (code) => {
            if (code === 0) {
              try {
                const parsedOutput = JSON.parse(bulkSummaryOutput);
                const { summaries } = parsedOutput;
                const summariesWithFilenames = summaries.map((summary, index) => ({
                  filename: files[index].originalname, // Add the original filename here
                  summary: summary.summary,
                  total_pages: summary.total_pages,
                }));
                res.status(200).json({ summaries: summariesWithFilenames });
              } catch (error) {
                console.error('Error parsing bulk summary output:', error);
                res.status(500).json({ error: 'Error parsing bulk summary output' });
              }
            } else {
              console.error('Bulk summary script failed with code:', code);
              res.status(500).json({ error: 'Bulk summary script failed' });
            }
          });
        
          bulkSummaryProcess.on('error', (error) => {
            console.error('An error occurred while running the bulk summary script:', error);
            res.status(500).json({ error: 'An error occurred while running the bulk summary script' });
          });
        } else if (selectedScript === 'process.py') {
          const processScriptPath = path.join(scriptsDir, 'process.py');
          const myenvPath = path.join(scriptsDir, 'myenv', 'bin', 'python3');
          const processProcess = spawn(myenvPath, [processScriptPath, batchDirectory, selectedModel]);

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
                const processedResults = JSON.parse(processOutput);
                res.status(200).json({ results: processedResults });
              } catch (error) {
                console.error('Error parsing process output:', error);
                res.status(500).json({ error: 'Error parsing process output', errorDetails: error.message });
              }
            } else {
              console.error('Process script failed with code:', code);
              res.status(500).json({ error: 'Process script failed', errorDetails: processOutput });
            }
          });
        } else if (selectedScript === 'toc.py') {
          const file = files[0];
          const tempOutputPath = `${batchDirectory}${file.filename}.json`;

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
        
                // Ensure the structure sent to the client matches expected format
                if (Array.isArray(tocData)) {
                  const responseData = {
                    tocData,
                    filePath: `uploads/${file.filename}` // Assuming the PDF is accessible via this path
                  };
                  res.status(200).json(responseData);
                } else {
                  console.error('Invalid TOC data format');
                  res.status(500).json({ error: 'Invalid TOC data format' });
                }
              } catch (error) {
                console.error('Error parsing TOC output:', error);
                res.status(500).json({ error: 'Error parsing TOC output', details: error.message });
              }
            } else {
              console.error('TOC script failed with code:', code);
              res.status(500).json({ error: 'TOC script failed' });
            }
          });
        } else if (selectedScript === 'entity.py') {
          const file = files[0];
          const tempOutputPath = `${batchDirectory}${file.filename}.json`;

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
          res.status(400).json({ error: 'Invalid script selected' });
        }
      } catch (error) {
        console.error('OCR script failed:', error);
        res.status(500).json({ error: 'OCR script failed' });
      }
    });
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}
import { NextApiRequest, NextApiResponse } from 'next';
import multer from 'multer';
import { spawn } from 'child_process';
import fs from 'fs';
import path from 'path';

const upload = multer({ dest: '/tmp' });

export const config = {
  api: {
    bodyParser: false,
  },
};

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

      const tempFilePath = file.path;

      res.writeHead(200, {
        'Content-Type': 'text/plain',
        'Transfer-Encoding': 'chunked',
      });

      const pythonProcess = spawn('python3', ['pages/api/web.py', tempFilePath]);

      pythonProcess.stdout.on('data', (data) => {
        res.write(data);
      });

      pythonProcess.stderr.on('data', (data) => {
        console.error(`Script error: ${data}`);
      });

      pythonProcess.on('close', (code) => {
        res.end();
        fs.unlinkSync(tempFilePath);
      });
    });
  } else {
    res.status(405).json({ error: 'Method not allowed' });
  }
}
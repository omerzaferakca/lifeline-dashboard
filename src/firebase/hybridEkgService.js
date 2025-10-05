// Hybrid EKG Analysis Service - Cloud Run Backend Integration
import { 
  ref, 
  uploadBytes,
  uploadBytesResumable,
  getDownloadURL, 
  deleteObject
} from 'firebase/storage';
import { 
  collection, 
  doc, 
  addDoc, 
  getDoc, 
  getDocs, 
  updateDoc, 
  deleteDoc,
  query,
  where,
  orderBy,
  serverTimestamp
} from 'firebase/firestore';
import { storage, db } from './config';

// Hybrid backend endpoints
const HYBRID_ENDPOINTS = {
  ANALYZE_MANUAL: 'https://lifeline-backend-140492000519.europe-west1.run.app/analyze',
  BACKEND_STATUS: 'https://lifeline-backend-140492000519.europe-west1.run.app/health'
};

export const hybridEkgService = {
  // Check hybrid backend status
  async checkBackendStatus() {
    try {
      const response = await fetch(HYBRID_ENDPOINTS.BACKEND_STATUS);
      return await response.json();
    } catch (error) {
      console.error('Backend status check failed:', error);
      return { functions_status: 'error', backend_status: { status: 'unreachable' } };
    }
  },

  // Analyze EKG file using hybrid backend
  async analyzeEkgFile(filePath) {
    try {
      console.log('Hybrid EKG analysis starting:', filePath);
      
      const response = await fetch(HYBRID_ENDPOINTS.ANALYZE_MANUAL, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ file_path: filePath })
      });

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.status} ${response.statusText}`);
      }

      const result = await response.json();
      console.log('🔥 Hybrid analysis RAW response:', result);
      console.log('🔥 Response keys:', Object.keys(result));
      console.log('🔥 analysis_result keys:', result.analysis_result ? Object.keys(result.analysis_result) : 'NONE');
      console.log('🔥 display_signal type:', typeof result.analysis_result?.display_signal);
      console.log('🔥 display_signal is array?:', Array.isArray(result.analysis_result?.display_signal));
      
      return result;
    } catch (error) {
      console.error('Hybrid EKG analysis failed:', error);
      throw error;
    }
  },

  // Upload EKG file with hybrid analysis
  async uploadEkgFileWithAnalysis(file, patientId, doctorId, progressCallback = null) {
    try {
      console.log('Uploading EKG file for hybrid analysis:', file.name);
      
      // Generate unique file path
      const timestamp = new Date().toISOString().replace(/[:.]/g, '');
      const filePath = `ekg/${doctorId}/${patientId}/${timestamp}_${file.name}`;
      const storageRef = ref(storage, filePath);

      let snapshot;
      
      // Upload with progress tracking
      if (progressCallback) {
        await new Promise((resolve, reject) => {
          const uploadTask = uploadBytesResumable(storageRef, file);
          
          uploadTask.on('state_changed',
            (progress) => {
              const progressPercent = progress.bytesTransferred / progress.totalBytes;
              progressCallback(progressPercent * 0.3); // Upload is 30% of progress
            },
            (error) => {
              console.error('Upload error:', error);
              reject(error);
            },
            () => {
              progressCallback(0.3); // Upload completed
              resolve(uploadTask.snapshot);
            }
          );
        });
        snapshot = await getDownloadURL(storageRef);
      } else {
        snapshot = await uploadBytes(storageRef, file);
      }

      // Get download URL
      const downloadURL = await getDownloadURL(storageRef);
      console.log('File uploaded, starting analysis...');
      
      // Create analysis record first
      const analysisDoc = await addDoc(collection(db, 'analyses'), {
        patientId: patientId,
        doctorId: doctorId,
        fileName: file.name,
        originalFileName: file.name,
        storagePath: filePath,
        downloadURL: downloadURL,
        fileSize: file.size,
        fileType: file.type,
        uploadDate: serverTimestamp(),
        createdAt: serverTimestamp(),
        updatedAt: serverTimestamp(),
        analysisStatus: 'processing',
        processed: false,
        processedBy: 'hybrid_backend'
      });
      
      if (progressCallback) progressCallback(0.4); // 40%

      // Perform hybrid analysis
      try {
        const analysisResult = await this.analyzeEkgFile(filePath);
        
        if (progressCallback) progressCallback(0.8); // 80%
        
        if (analysisResult.success) {
          // Update analysis record with results
          await updateDoc(doc(db, 'analyses', analysisDoc.id), {
            analysisStatus: 'completed',
            analysisResult: analysisResult.analysis_result,
            heartRate: analysisResult.analysis_result?.clinical_metrics?.heart_rate_bpm || 0,
            riskLevel: analysisResult.analysis_result?.ai_summary?.risk_level || 'bilinmiyor',
            rhythmType: analysisResult.analysis_result?.ai_summary?.rhythm_type || 'bilinmiyor',
            processed: true,
            processedAt: serverTimestamp(),
            updatedAt: serverTimestamp()
          });
          
          if (progressCallback) progressCallback(1.0); // 100%
          
          console.log('Hybrid analysis completed successfully');
          
          return {
            success: true,
            id: analysisDoc.id,
            downloadURL: downloadURL,
            analysisResult: analysisResult.analysis_result
          };
        } else {
          // Analysis failed
          await updateDoc(doc(db, 'analyses', analysisDoc.id), {
            analysisStatus: 'failed',
            analysisError: analysisResult.error || 'Analysis failed',
            processed: false,
            processedAt: serverTimestamp(),
            updatedAt: serverTimestamp()
          });
          
          throw new Error(analysisResult.error || 'Analysis failed');
        }
      } catch (analysisError) {
        console.error('Analysis error:', analysisError);
        
        // Update record with error
        await updateDoc(doc(db, 'analyses', analysisDoc.id), {
          analysisStatus: 'failed',
          analysisError: analysisError.message || 'Analysis error',
          processed: false,
          processedAt: serverTimestamp(),
          updatedAt: serverTimestamp()
        });
        
        throw analysisError;
      }
      
    } catch (error) {
      console.error('Upload with analysis failed:', error);
      if (progressCallback) progressCallback(0);
      throw error;
    }
  },

  // Get patient EKG files (same as original)
  async getPatientEkgFiles(patientId) {
    try {
      console.log('Getting patient EKG files:', patientId);
      
      const q = query(
        collection(db, 'analyses'),
        where('patientId', '==', patientId),
        orderBy('uploadDate', 'desc')
      );
      
      const querySnapshot = await getDocs(q);
      const files = [];
      
      for (const doc of querySnapshot.docs) {
        const data = doc.data();
        files.push({
          id: doc.id,
          ...data,
          uploaded_at: data.uploadedAt?.toDate?.()?.toISOString() || data.uploadedAt
        });
      }
      
      console.log(`${files.length} EKG files found`);
      return files;
    } catch (error) {
      console.error('Error getting EKG files:', error);
      throw error;
    }
  },

  // Delete EKG file
  async deleteEkgFile(analysisId, storagePath) {
    try {
      console.log('Deleting EKG file:', analysisId);
      
      // Delete from Firestore
      const analysisRef = doc(db, 'analyses', analysisId);
      await deleteDoc(analysisRef);
      
      // Delete from Storage
      if (storagePath) {
        const storageRef = ref(storage, storagePath);
        await deleteObject(storageRef);
      }
      
      console.log('EKG file deleted successfully');
      return { success: true };
    } catch (error) {
      console.error('Error deleting EKG file:', error);
      return {
        success: false,
        error: error.message || 'Delete error'
      };
    }
  },

  // Validate EKG file
  validateEkgFile(file) {
    const maxSize = 100 * 1024 * 1024; // 100MB
    const allowedExtensions = ['.csv', '.txt', '.dat'];
    
    if (file.size > maxSize) {
      throw new Error('Dosya boyutu 100MB\'yi aşamaz');
    }
    
    const extension = file.name.toLowerCase().slice(file.name.lastIndexOf('.'));
    if (!allowedExtensions.includes(extension)) {
      throw new Error('Desteklenen formatlar: CSV, TXT, DAT');
    }
    
    return true;
  }
};

// Named exports
export const uploadEkgFileWithAnalysis = hybridEkgService.uploadEkgFileWithAnalysis.bind(hybridEkgService);
export const analyzeEkgFile = hybridEkgService.analyzeEkgFile.bind(hybridEkgService);
export const checkBackendStatus = hybridEkgService.checkBackendStatus.bind(hybridEkgService);
export const getPatientEkgFiles = hybridEkgService.getPatientEkgFiles.bind(hybridEkgService);
export const deleteEkgFile = hybridEkgService.deleteEkgFile.bind(hybridEkgService);
export const validateEkgFile = hybridEkgService.validateEkgFile.bind(hybridEkgService);

export default hybridEkgService;
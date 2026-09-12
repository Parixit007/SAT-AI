import { useRef, useState } from "react";
import { uploadImages, type UploadResponse } from "../api/client";

interface Props {
  onUploaded: (result: UploadResponse) => void;
}

export function UploadPanel({ onUploaded }: Props) {
  const [files, setFiles] = useState<File[]>([]);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList) return;
    setFiles(Array.from(fileList));
    setError(null);
  };

  const handleUpload = async () => {
    if (files.length === 0) return;
    setUploading(true);
    setError(null);
    try {
      const result = await uploadImages(files);
      onUploaded(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-block">
      <div
        className="dropzone"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          handleFiles(e.dataTransfer.files);
        }}
      >
        {files.length === 0 ? (
          <span>Drop one image, or a pair — PNG, JPEG, TIFF</span>
        ) : (
          <ul className="file-list">
            {files.map((f) => (
              <li key={f.name}>{f.name}</li>
            ))}
          </ul>
        )}
        <input
          ref={inputRef}
          type="file"
          accept="image/png,image/jpeg,image/tiff"
          multiple
          hidden
          onChange={(e) => handleFiles(e.target.files)}
        />
      </div>
      <button disabled={files.length === 0 || uploading} onClick={handleUpload}>
        {uploading ? "Uploading..." : "Upload"}
      </button>
      {error && <p className="error-text">{error}</p>}
    </div>
  );
}

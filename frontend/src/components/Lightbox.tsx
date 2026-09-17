import { AnimatePresence, motion } from "framer-motion";
import { useEffect, useState } from "react";
import { XIcon } from "./icons";

// Wraps any image so clicking it opens an enlarged view over a backdrop -- used for evidence
// renders and the grounding overlay's base image, where the inline size is too small to read
// exact detail (e.g. a small detection box) but full detail matters for an evidence-grounded app.
export function LightboxImage({ src, alt, className }: { src: string; alt: string; className?: string }) {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open]);

  return (
    <>
      <img
        src={src}
        alt={alt}
        className={className}
        onClick={() => setOpen(true)}
        role="button"
        tabIndex={0}
        onKeyDown={(e) => {
          if (e.key === "Enter" || e.key === " ") {
            e.preventDefault();
            setOpen(true);
          }
        }}
      />
      <AnimatePresence>
        {open && (
          <motion.div
            className="lightbox-backdrop"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.16 }}
            onClick={() => setOpen(false)}
          >
            <motion.img
              src={src}
              alt={alt}
              className="lightbox-image"
              initial={{ opacity: 0, scale: 0.94 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.96 }}
              transition={{ duration: 0.18, ease: [0.2, 0.8, 0.2, 1] }}
              onClick={(e) => e.stopPropagation()}
            />
            <button className="btn btn-icon lightbox-close" onClick={() => setOpen(false)} title="Close" aria-label="Close">
              <XIcon size={18} />
            </button>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

import { motion } from "framer-motion";
import type { ToolSpecOut } from "../api/client";
import { EXAMPLE_PROMPT, TOOL_ICON, TOOL_LABEL } from "../toolMeta";

// The homepage: replaces a bare "attach imagery or open the map" empty state with a real landing
// section, so the pipeline's actual breadth is visible before a query is ever typed. Tools come
// from the live registry (GET /api/tools) -- this can never list a tool that doesn't exist.
export function CapabilitiesGallery({ tools, onPick }: { tools: ToolSpecOut[]; onPick: (tool: ToolSpecOut) => void }) {
  return (
    <div className="homepage">
      <div className="homepage-hero">
        <span className="homepage-eyebrow">SatQuery AI · 7 specialists, one query</span>
        <h1>Query the pipeline</h1>
        <p>
          Route a query to the right remote-sensing specialist automatically, or pick one yourself.
          Attach imagery, drop a pin, and ask.
        </p>
      </div>

      <div className="capability-grid">
        {tools.map((tool, i) => (
          <motion.button
            key={tool.name}
            className="capability-card"
            onClick={() => onPick(tool)}
            initial={{ opacity: 0, y: 14 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, delay: i * 0.05, ease: [0.2, 0.8, 0.2, 1] }}
            whileHover={{ y: -3 }}
            whileTap={{ scale: 0.98 }}
          >
            <span className="capability-icon" aria-hidden="true">
              {TOOL_ICON[tool.name] ?? "🛰️"}
            </span>
            <span className="capability-name">{TOOL_LABEL[tool.name] ?? tool.name}</span>
            <span className="capability-desc">{tool.description}</span>
            <span className="capability-prompt data-value">
              {EXAMPLE_PROMPT[tool.name] ?? "Try this specialist"}
            </span>
          </motion.button>
        ))}
      </div>
    </div>
  );
}

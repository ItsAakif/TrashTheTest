import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import type { NotesGenerationConfig } from "@/lib/api-services";

interface NotesGenerationConfigDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  value: NotesGenerationConfig;
  onChange: (value: NotesGenerationConfig) => void;
  recommendation?: string;
  /** Optional confirm action (e.g. start generation with these settings). */
  onConfirm?: () => void;
  confirmLabel?: string;
}

const labelClass = "block text-sm font-medium text-stone-300 mb-2";

// Mode labels are user-facing; the values are the wire enum the backend
// NotesGenerationConfig accepts (mode: quick | detailed | exam | memory |
// complete).  Field names are exactly the API contract — never invented.
const MODE_OPTIONS: Array<{ value: NonNullable<NotesGenerationConfig["mode"]>; label: string }> = [
  { value: "complete", label: "Complete Study Notes" },
  { value: "quick", label: "Quick Summary" },
  { value: "detailed", label: "Detailed Explanation" },
  { value: "exam", label: "Exam Revision" },
  { value: "memory", label: "Memory Boost" },
];

const DETAIL_OPTIONS: Array<{ value: NonNullable<NotesGenerationConfig["detail"]>; label: string }> = [
  { value: "concise", label: "Concise" },
  { value: "balanced", label: "Balanced" },
  { value: "comprehensive", label: "Comprehensive" },
];

const SCOPE_OPTIONS: Array<{ value: NonNullable<NotesGenerationConfig["scope"]>; label: string }> = [
  { value: "entire", label: "Entire project" },
  { value: "selected", label: "Selected documents" },
];

export function NotesGenerationConfigDialog({
  open,
  onOpenChange,
  value,
  onChange,
  recommendation,
  onConfirm,
  confirmLabel = "Generate notes",
}: NotesGenerationConfigDialogProps) {
  const update = <K extends keyof NotesGenerationConfig>(key: K, next: NotesGenerationConfig[K]) => {
    onChange({ ...value, [key]: next });
  };

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="max-w-xl border-stone-700 bg-[#131313] text-stone-100">
        <DialogHeader>
          <DialogTitle>Notes settings</DialogTitle>
          <DialogDescription className="text-stone-400">
            Configure this run only. Uploaded material remains the primary source.
          </DialogDescription>
        </DialogHeader>

        <div className="grid gap-5 py-2">
          {recommendation ? (
            <p className="rounded-lg border border-stone-700 bg-stone-900 p-3 text-xs text-stone-300">
              {recommendation}
            </p>
          ) : null}

          <div>
            <label className={labelClass} htmlFor="notes-mode">Mode</label>
            <select
              id="notes-mode"
              value={value.mode ?? "detailed"}
              onChange={(event) => update("mode", event.target.value as NotesGenerationConfig["mode"])}
              className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-2 text-sm"
            >
              {MODE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className={labelClass} htmlFor="notes-detail">Detail level</label>
            <select
              id="notes-detail"
              value={value.detail ?? "balanced"}
              onChange={(event) => update("detail", event.target.value as NotesGenerationConfig["detail"])}
              className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-2 text-sm"
            >
              {DETAIL_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>

          <div>
            <label className={labelClass} htmlFor="notes-scope">Source scope</label>
            <select
              id="notes-scope"
              value={value.scope ?? "entire"}
              onChange={(event) => update("scope", event.target.value as NotesGenerationConfig["scope"])}
              className="w-full rounded-lg border border-stone-700 bg-stone-900 px-3 py-2 text-sm"
            >
              {SCOPE_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>

          <div className="space-y-3 text-sm text-stone-300">
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={Boolean(value.web)}
                onChange={(event) => update("web", event.target.checked)}
              />
              Use web research to improve explanations
            </label>
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={Boolean(value.exam_focus)}
                onChange={(event) => update("exam_focus", event.target.checked)}
              />
              Emphasise exam/revision information
            </label>
            <label className="flex items-center gap-3">
              <input
                type="checkbox"
                checked={Boolean(value.memory_aids)}
                onChange={(event) => update("memory_aids", event.target.checked)}
              />
              Add mnemonics and memory techniques
            </label>
          </div>
        </div>

        <DialogFooter className="gap-2">
          <button
            type="button"
            onClick={() => onOpenChange(false)}
            className="rounded-lg px-5 py-2 text-sm font-semibold"
            style={{ color: "#e7e5e4", border: "1px solid rgba(72,72,72,0.3)" }}
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={() => {
              onConfirm?.();
              onOpenChange(false);
            }}
            className="rounded-lg bg-stone-100 px-5 py-2 text-sm font-semibold text-stone-900"
          >
            {confirmLabel}
          </button>
        </DialogFooter>
      </DialogContent>
    </Dialog>
  );
}

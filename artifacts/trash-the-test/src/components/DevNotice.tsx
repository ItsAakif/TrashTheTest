export default function DevNotice() {
  return (
    <div
      className="px-4 py-3 rounded-xl text-sm mb-4"
      style={{
        backgroundColor: "rgba(42,62,70,0.2)",
        border: "1px solid rgba(140,161,170,0.15)",
        color: "#8ca1aa",
      }}
    >
      <div className="flex items-center gap-2">
        <span
          className="material-symbols-outlined text-base"
          style={{ fontSize: "18px" }}
        >
          construction
        </span>
        <p>
          <strong>Backend in development.</strong> This feature needs a running
          backend server. Some data shown is placeholder content.
        </p>
      </div>
    </div>
  );
}

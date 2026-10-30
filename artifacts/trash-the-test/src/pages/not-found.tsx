import { Card, CardContent } from "@/components/ui/card";
import { AlertCircle } from "lucide-react";

export default function NotFound() {
  return (
    <div
      className="min-h-screen w-full flex items-center justify-center"
      style={{ backgroundColor: "#0e0e0e" }}
    >
      <Card
        className="w-full max-w-md mx-4 border"
        style={{
          backgroundColor: "#131313",
          borderColor: "rgba(72,72,72,0.15)",
        }}
      >
        <CardContent className="pt-6">
          <div className="flex mb-4 gap-2">
            <AlertCircle className="h-8 w-8 text-red-500" />
            <h1 className="text-2xl font-bold" style={{ color: "#e7e5e4" }}>
              404 Page Not Found
            </h1>
          </div>

          <p className="mt-4 text-sm" style={{ color: "#acabaa" }}>
            Did you forget to add the page to the router?
          </p>
        </CardContent>
      </Card>
    </div>
  );
}

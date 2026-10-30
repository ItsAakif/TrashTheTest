import { useQuery } from "@tanstack/react-query";
import Layout from "@/components/Layout";
import { SpinnerIcon } from "@/components/icons";
import { resolvePublicWebsite, publicWebsiteUrl } from "@/lib/api-services";

interface PublicWebsiteProps {
  params?: { slug?: string };
}

/**
 * Domain-agnostic public route: resolves PUBLISHED websites only.
 * Drafts/private get an honest not-found state; private docs never leak
 * (only the persisted sanitised HTML is rendered, sandboxed).
 */
export default function PublicWebsite({ params }: PublicWebsiteProps) {
  const slug = params?.slug ?? null;

  const {
    data: website,
    isLoading,
    isError,
  } = useQuery({
    queryKey: ["public-website", slug],
    queryFn: () => resolvePublicWebsite(slug as string),
    enabled: !!slug,
    retry: false,
  });

  return (
    <Layout activeNav="publish">
      <div className="pb-20 md:pb-8 px-6 md:px-12 max-w-5xl mx-auto pt-10">
        {!slug && (
          <p style={{ color: "#acabaa" }}>No website slug provided.</p>
        )}
        {slug && isLoading && (
          <div className="flex flex-col items-center justify-center h-96 rounded-xl" style={{ backgroundColor: "#1f2020" }}>
            <div className="mb-4">
              <SpinnerIcon size={32} color="#acabaa" />
            </div>
            <p style={{ color: "#acabaa" }}>Loading shared study website...</p>
          </div>
        )}
        {slug && !isLoading && (isError || !website) && (
          <div className="text-center py-16">
            <h1 className="text-2xl font-bold mb-3" style={{ color: "#e7e5e4" }}>
              Website not found
            </h1>
            <p className="max-w-md mx-auto" style={{ color: "#acabaa" }}>
              This link is invalid, or the website is a private draft that has
              not been published.
            </p>
          </div>
        )}
        {slug && website && (
          <div className="space-y-4">
            <header>
              <h1 className="text-3xl font-extrabold tracking-tight mb-2" style={{ color: "#e7e5e4" }}>
                {website.title}
              </h1>
              <p className="text-sm" style={{ color: "#acabaa" }}>
                Shared study website · {website.content_type}
              </p>
            </header>
            <div className="rounded-xl overflow-hidden" style={{ border: "1px solid rgba(72,72,72,0.3)" }}>
              <iframe
                src={publicWebsiteUrl(slug)}
                className="w-full"
                style={{ height: "80vh" }}
                title={website.title}
                sandbox=""
              />
            </div>
          </div>
        )}
      </div>
    </Layout>
  );
}

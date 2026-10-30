import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  citationLabel,
  generateAnswer,
  generateExplanation,
  generateMCQ,
  createStudyWebsite,
  publishWebsite,
  unpublishWebsite,
  setWebsiteVisibility,
  resolvePublicWebsite,
  publicWebsiteUrl,
  uploadDocumentsMulti,
  readNotes,
  readQuiz,
} from '../lib/api-services';

// Mock fetch globally
global.fetch = vi.fn();

describe('Study website pipeline (frontend states)', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('citationLabel', () => {
    it('renders chunk span/page/method, not a bare id', () => {
      const label = citationLabel(
        { chunk_id: 'abcdef123456', page: 4, method: 'ppocr', span: 'chlorophyll absorbs light' },
        0
      );
      expect(label).toContain('abcdef12');
      expect(label).toContain('p.4');
      expect(label).toContain('ppocr');
      expect(label).not.toBe('abcdef123456');
    });

    it('falls back to a numbered source when no provenance fields exist', () => {
      expect(citationLabel({ chunk_id: 'x'.repeat(20) }, 2)).toContain('Source 3');
    });
  });

  describe('generateAnswer', () => {
    it('requires a query (loading state never fires without one)', async () => {
      await expect(generateAnswer({ projectId: 'p1', query: '   ' })).rejects.toThrow(
        /Query is required/
      );
      expect(fetch).not.toHaveBeenCalled();
    });

    it('posts marks-aware payload and surfaces the word band', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({
          projectId: 'p1',
          mode: 'answer',
          content: { answer_text: 'word '.repeat(160) },
          marks: 5,
          word_band: { min_words: 150, max_words: 350 },
        }),
      } as Response);

      const result = await generateAnswer({ projectId: 'p1', query: 'What is X?', marks: 5 });
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/generation/answer'),
        expect.objectContaining({ method: 'POST' })
      );
      expect(result.word_band).toEqual({ min_words: 150, max_words: 350 });
    });

    it('surfaces server errors (error state)', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 400,
        statusText: 'Bad Request',
        json: async () => ({ detail: 'marks must be between 1 and 100' }),
      } as unknown as Response);
      await expect(generateAnswer({ projectId: 'p1', query: 'Q?', marks: 0 })).rejects.toThrow(
        /marks must be between/
      );
    });
  });

  describe('generateExplanation', () => {
    it('requires a query', async () => {
      await expect(generateExplanation({ projectId: 'p1', query: '' })).rejects.toThrow(
        /Query is required/
      );
      expect(fetch).not.toHaveBeenCalled();
    });

    it('posts to /api/generation/explanation', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ projectId: 'p1', mode: 'explanation', content: {} }),
      } as Response);
      await generateExplanation({ projectId: 'p1', query: 'photosynthesis' });
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/generation/explanation'),
        expect.any(Object)
      );
    });
  });

  describe('createStudyWebsite', () => {
    it('uses /api/website/generate with content_type/title/structured_content', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ slug: 'bio-abc123', title: 'Bio' }),
      } as Response);

      await createStudyWebsite({
        projectId: 'p1',
        contentType: 'mixed',
        title: 'Bio',
        structuredContent: { sections: [] },
      });

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/website/generate'),
        expect.objectContaining({ method: 'POST' })
      );
      const body = JSON.parse(vi.mocked(fetch).mock.calls[0][1].body as string);
      expect(body.project_id).toBe('p1');
      expect(body.content_type).toBe('mixed');
      expect(body.title).toBe('Bio');
      expect(body).not.toHaveProperty('projectId');
      expect(body).not.toHaveProperty('contentType');
      expect(body.structured_content).toEqual({ sections: [] });
      // Must not hit the content-only endpoint.
      expect(vi.mocked(fetch).mock.calls[0][0] as string).not.toContain('/api/generation/website');
    });
  });

  describe('publish / visibility states', () => {
    it('publish posts the slug and returns preview + public urls', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'published', slug: 's1', preview_url: 'p', public_url: 'u' }),
      } as Response);
      const out = await publishWebsite('s1');
      expect(out.status).toBe('published');
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/website/publish'),
        expect.any(Object)
      );
    });

    it('unpublish returns the draft state', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'draft', slug: 's1' }),
      } as Response);
      expect((await unpublishWebsite('s1')).status).toBe('draft');
    });

    it('visibility toggle posts public/private', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ status: 'ok', slug: 's1', visibility: 'public' }),
      } as Response);
      const out = await setWebsiteVisibility('s1', 'public');
      expect(out.visibility).toBe('public');
    });

    it('public resolve hits the published-only route; share url is domain-agnostic', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ slug: 's1', is_published: true }),
      } as Response);
      await resolvePublicWebsite('s1');
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/website/public/s1'),
        expect.any(Object)
      );
      expect(publicWebsiteUrl('s1')).toContain('/api/website/public/s1/html');
    });

    it('private/draft public resolve surfaces 404 (error state, never leaked)', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 404,
        statusText: 'Not Found',
        json: async () => ({ detail: 'Website not found' }),
      } as unknown as Response);
      await expect(resolvePublicWebsite('draft-1')).rejects.toThrow();
    });
  });

  describe('uploadDocumentsMulti', () => {
    it('posts FormData to /api/documents/upload/multi and returns per-file jobs', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => [
          { fileName: 'a.pdf', documentId: 'd1', jobId: 'j1', status: 'pending', error: null },
          { fileName: 'b.pdf', documentId: null, jobId: null, status: 'failed', error: 'bad type' },
        ],
      } as Response);

      const out = await uploadDocumentsMulti(
        [new File(['x'], 'a.pdf'), new File(['y'], 'b.pdf')],
        'p1'
      );
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/documents/upload/multi'),
        expect.any(Object)
      );
      expect(out).toHaveLength(2);
      expect(out[1].status).toBe('failed');
    });
  });

  describe('reads (loading/empty/error)', () => {
    it('readNotes returns the not_generated empty state', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ projectId: 'p1', state: 'not_generated', notes: [], total: 0 }),
      } as Response);
      const out = await readNotes('p1');
      expect(out.state).toBe('not_generated');
    });

    it('readQuiz propagates errors for the error state', async () => {
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'));
      await expect(readQuiz('p1')).rejects.toThrow('Network error');
    });

    it('generateMCQ posts to the strict endpoint', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ mode: 'mcq' }),
      } as Response);
      await generateMCQ({ projectId: 'p1' });
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/generation/mcq'),
        expect.any(Object)
      );
    });
  });
});

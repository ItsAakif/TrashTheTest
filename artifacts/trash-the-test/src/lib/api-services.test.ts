import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { getProjects, getFlashcards, getProject, createProject, login, register, getCurrentUser, pollJobUntilDone, runHierarchicalNotes, JobRunError } from '../lib/api-services';
  import {
  generateNotes,
  generateFlashcards,
  generateQuiz,
  generateSummary,
  generateWebsite,
  generateMCQ,
  generatePractice,
  fetchDocumentBytes,
  searchContent,
  startStudySession,
  sendTutorMessage,
  submitStudyResponse,
  completeStudySession,
  saveProviderSettings,
  testProviderConnection,
  getProviderStatus,
  listProviderModels,
  removeProviderSettings,
  createStudyWebsite,
} from '../lib/api-services';

// Mock fetch globally
global.fetch = vi.fn();

describe('API Services', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(() => {
    vi.resetAllMocks();
  });

  describe('getProjects', () => {
    it('should return empty array on successful request with no projects', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ projects: [], total: 0 }),
      } as Response);

      const result = await getProjects();

      expect(result).toEqual([]);
      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/projects'),
        expect.any(Object)
      );
    });

    it('should normalise the backend shape (title/created_at) to the UI shape (name/createdAt)', async () => {
      // Backend contract: {id, title, description, status, progress,
      // created_at, updated_at}. UI contract: {id, name, ..., createdAt}.
      const backendProjects = [
        { id: '1', title: 'Data Structures', description: 'DS', status: 'processing', progress: 0, created_at: '2026-09-18T00:00:00', updated_at: '2026-09-18T00:00:00' },
      ];

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ projects: backendProjects, total: 1 }),
      } as Response);

      const result = await getProjects();

      expect(result).toEqual([
        {
          id: '1',
          name: 'Data Structures',
          description: 'DS',
          status: 'processing',
          progress: 0,
          createdAt: '2026-09-18T00:00:00',
          updatedAt: '2026-09-18T00:00:00',
          documentCount: undefined,
          lastActivity: undefined,
        },
      ]);
    });

    it('should return empty array and log error on network failure', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Network error'));

      const result = await getProjects();

      expect(result).toEqual([]);
      expect(consoleSpy).toHaveBeenCalledWith(
        expect.stringContaining('[API Error]'),
        expect.any(Error)
      );
      consoleSpy.mockRestore();
    });

    it('should return empty array when response is not ok', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 500,
      } as Response);

      const result = await getProjects();

      expect(result).toEqual([]);
      consoleSpy.mockRestore();
    });
  });

  describe('getFlashcards', () => {
    it('should return empty deck when no flashcards exist', async () => {
      const emptyDeck = {
        id: 'deck-1',
        projectId: 'project-1',
        name: 'Deck',
        cards: [],
        cardCount: 0,
      };
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => emptyDeck,
      } as Response);

      const result = await getFlashcards({ projectId: 'project-1' });

      expect(result).toEqual(emptyDeck);
    });

    it('should return flashcard deck on success', async () => {
      const mockDeck = {
        id: 'deck-1',
        projectId: 'project-1',
        name: 'Deck',
        cards: [
          { id: '1', front: 'Question 1', back: 'Answer 1' },
          { id: '2', front: 'Question 2', back: 'Answer 2' },
        ],
        cardCount: 2,
      };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => mockDeck,
      } as Response);

      const result = await getFlashcards({ projectId: 'project-1' });

      expect(result).toEqual(mockDeck);
    });

    it('should return empty deck on error', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Failed to fetch'));

      const result = await getFlashcards({ projectId: 'project-1' });

      expect(result).toEqual({
        id: 'deck_empty',
        projectId: 'project-1',
        name: 'No Flashcards',
        cards: [],
        cardCount: 0,
      });
      expect(consoleSpy).toHaveBeenCalled();
      consoleSpy.mockRestore();
    });
  });

  describe('getProject', () => {
    it('should return null when project not found', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ project: null }),
      } as Response);

      const result = await getProject('nonexistent-id');

      expect(result).toBeNull();
    });

    it('should normalise a single backend project to the UI shape', async () => {
      const backendProject = { id: '1', title: 'Test', description: null, status: 'processing', progress: 100, created_at: '2026-09-18T00:00:00', updated_at: '2026-09-18T00:00:00' };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => backendProject,
      } as Response);

      const result = await getProject('1');

      expect(result).toEqual({
        id: '1',
        name: 'Test',
        description: undefined,
        status: 'processing',
        progress: 100,
        createdAt: '2026-09-18T00:00:00',
        updatedAt: '2026-09-18T00:00:00',
        documentCount: undefined,
        lastActivity: undefined,
      });
    });

    it('should return null on error', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      vi.mocked(fetch).mockRejectedValueOnce(new Error('Failed'));

      const result = await getProject('1');

      expect(result).toBeNull();
      consoleSpy.mockRestore();
    });
  });

  describe('createProject', () => {
    it('should POST title+name and return the normalised project', async () => {
      const backendProject = { id: '9', title: 'Data Structures', description: 'DS', status: 'processing', progress: 0, created_at: '2026-09-18T00:00:00', updated_at: '2026-09-18T00:00:00' };

      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => backendProject,
      } as Response);

      const result = await createProject({ name: 'Data Structures', description: 'DS' });

      expect(fetch).toHaveBeenCalledWith(
        expect.stringContaining('/api/projects'),
        expect.objectContaining({ method: 'POST' })
      );
      const init = vi.mocked(fetch).mock.calls[0][1] as RequestInit | undefined;
      const body = JSON.parse(String(init?.body)) as { title?: string };
      expect(body.title).toBe('Data Structures');
      expect(result.name).toBe('Data Structures');
      expect(result.id).toBe('9');
    });

    it('should throw the server error when creation fails', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 422,
        json: async () => ({ detail: 'title (or name) is required' }),
      } as unknown as Response);

      await expect(createProject({ name: '  ' })).rejects.toThrow('title (or name) is required');
      consoleSpy.mockRestore();
    });
  });

  describe('auth helpers', () => {
    // Self-contained storage stub so these run under a node env too.
    const store: Record<string, string> = {};
    beforeEach(() => {
      for (const key of Object.keys(store)) delete store[key];
      Object.defineProperty(globalThis, 'localStorage', {
        value: {
          getItem: (k: string) => (k in store ? store[k] : null),
          setItem: (k: string, v: string) => { store[k] = String(v); },
          removeItem: (k: string) => { delete store[k]; },
        },
        configurable: true,
      });
    });

    it('login normalises email and omits a stale Authorization header', async () => {
      store['trash_the_test_token'] = 'stale-token';
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ access_token: 'new-token', user: { id: '1', email: 'a@b.c', name: 'A' } }),
      } as Response);

      const result = await login('  Mixed@Example.COM ', 'pw');

      expect(result.token).toBe('new-token');
      const [url, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit?];
      expect(url).toContain('/api/auth/login');
      expect(JSON.parse(String(init?.body)).email).toBe('mixed@example.com');
      expect(new Headers(init?.headers).get('authorization')).toBeNull();
    });

    it('login clears a stale token when credentials are rejected', async () => {
      store['trash_the_test_token'] = 'stale-token';
      store['user'] = '{"id":"1"}';
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ detail: 'Invalid email or password' }),
      } as unknown as Response);

      await expect(login('a@b.c', 'wrong')).rejects.toThrow('Invalid email or password');
      expect(store['trash_the_test_token']).toBeUndefined();
      expect(store['user']).toBeUndefined();
    });

    it('register normalises email and omits the Authorization header', async () => {
      store['trash_the_test_token'] = 'stale-token';
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ access_token: 'tok', user: { id: '2', email: 'n@x.y', name: 'N' } }),
      } as Response);

      await register('  New@Example.COM ', 'password123', 'N');

      const [url, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit?];
      expect(url).toContain('/api/auth/register');
      expect(JSON.parse(String(init?.body)).email).toBe('new@example.com');
      expect(new Headers(init?.headers).get('authorization')).toBeNull();
    });

    it('getCurrentUser renews an expired access token via refresh instead of logging out', async () => {
      store['trash_the_test_token'] = 'expired-access';
      store['trash_the_test_refresh_token'] = 'valid-refresh';
      // /me rejects the expired access token...
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ detail: 'Not authenticated' }),
      } as unknown as Response);
      // ...refresh succeeds without an Authorization header...
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ access_token: 'fresh-access', token_type: 'bearer' }),
      } as Response);
      // ...and /me succeeds on retry.
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => ({ id: '7', email: 'u@x.y', name: 'U' }),
      } as Response);

      const user = await getCurrentUser();

      expect(user).toEqual({ id: '7', email: 'u@x.y', name: 'U' });
      expect(store['trash_the_test_token']).toBe('fresh-access');
      const refreshCall = vi.mocked(fetch).mock.calls[1] as [string, RequestInit?];
      expect(String(refreshCall[0])).toContain('/api/auth/refresh');
      expect(JSON.parse(String(refreshCall[1]?.body)).refresh_token).toBe('valid-refresh');
    });

    it('getCurrentUser signs out only when refresh also fails', async () => {
      store['trash_the_test_token'] = 'expired-access';
      store['trash_the_test_refresh_token'] = 'dead-refresh';
      store['user'] = '{"id":"7"}';
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ detail: 'Not authenticated' }),
      } as unknown as Response);
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: false,
        status: 401,
        json: async () => ({ detail: 'Invalid refresh' }),
      } as unknown as Response);

      expect(await getCurrentUser()).toBeNull();
      expect(store['trash_the_test_token']).toBeUndefined();
      expect(store['trash_the_test_refresh_token']).toBeUndefined();
      expect(store['user']).toBeUndefined();
    });

    it('shares one silent refresh across concurrent 401 reads and retries each once', async () => {
      store['trash_the_test_token'] = 'expired-access';
      store['trash_the_test_refresh_token'] = 'valid-refresh';

      let refreshCalls = 0;
      let protectedCalls = 0;
      let releaseRefresh: (() => void) | undefined;
      let markRefreshStarted: (() => void) | undefined;
      const refreshGate = new Promise<void>((resolve) => { releaseRefresh = resolve; });
      const refreshStarted = new Promise<void>((resolve) => { markRefreshStarted = resolve; });

      vi.mocked(fetch).mockImplementation(async (input) => {
        const url = String(input);
        if (url.includes('/api/auth/refresh')) {
          refreshCalls += 1;
          markRefreshStarted?.();
          await refreshGate;
          return {
            ok: true,
            status: 200,
            json: async () => ({ access_token: 'fresh-access', refresh_token: 'rotated-refresh' }),
          } as Response;
        }

        protectedCalls += 1;
        if (protectedCalls <= 2) {
          return {
            ok: false,
            status: 401,
            json: async () => ({ detail: 'Authentication required' }),
          } as Response;
        }
        if (url.includes('/api/auth/me')) {
          return {
            ok: true,
            status: 200,
            json: async () => ({ id: '7', email: 'u@x.y', name: 'U' }),
          } as Response;
        }
        return {
          ok: true,
          status: 200,
          json: async () => ({ projects: [], total: 0 }),
        } as Response;
      });

      const pending = Promise.all([getProjects(), getCurrentUser()]);
      await refreshStarted;
      expect(refreshCalls).toBe(1);
      releaseRefresh?.();

      const [projects, user] = await pending;
      expect(projects).toEqual([]);
      expect(user?.id).toBe('7');
      expect(refreshCalls).toBe(1);
      expect(store['trash_the_test_token']).toBe('fresh-access');
      expect(store['trash_the_test_refresh_token']).toBe('rotated-refresh');

      const protectedRetries = vi.mocked(fetch).mock.calls
        .filter(([url]) => !String(url).includes('/api/auth/refresh'))
        .slice(2)
        .map(([, init]) => new Headers(init?.headers).get('authorization'));
      expect(protectedRetries).toEqual([
        'Bearer fresh-access',
        'Bearer fresh-access',
      ]);
    });

    it('keeps renewed storage when the original request is rejected again', async () => {
      const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});
      store['trash_the_test_token'] = 'expired-access';
      store['trash_the_test_refresh_token'] = 'valid-refresh';
      vi.mocked(fetch)
        .mockResolvedValueOnce({
          ok: false,
          status: 401,
          json: async () => ({ detail: 'Authentication required' }),
        } as unknown as Response)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => ({ access_token: 'fresh-access' }),
        } as Response)
        .mockResolvedValueOnce({
          ok: false,
          status: 401,
          json: async () => ({ detail: 'Authentication required' }),
        } as unknown as Response);

      expect(await getProjects()).toEqual([]);
      expect(store['trash_the_test_token']).toBe('fresh-access');
      expect(store['trash_the_test_refresh_token']).toBe('valid-refresh');
      expect(vi.mocked(fetch)).toHaveBeenCalledTimes(3);
      consoleSpy.mockRestore();
    });
  });

  describe('academic job progress', () => {
    it('normalizes snake_case progress and hydrates terminal failure details', async () => {
      vi.mocked(fetch)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => ({
            job_id: 'job-1',
            status: 'failed',
            progress: 94,
            stage: 'persist',
            stage_message: 'persisted',
            created_at: '2026-09-25T00:00:00Z',
          }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => ({
            id: 'job-1',
            job_type: 'notes_hierarchical',
            status: 'failed',
            progress: 94,
            stage: 'persist',
            stage_message: 'persisted',
            error_message: 'PDF storage failed',
            output_data: { artifact_id: null },
            created_at: '2026-09-25T00:00:00Z',
          }),
        } as Response);

      const job = await pollJobUntilDone('job-1', { intervalMs: 1 });
      expect(job.id).toBe('job-1');
      expect(job.jobType).toBe('notes_hierarchical');
      expect(job.stageMessage).toBe('persisted');
      expect(job.errorMessage).toBe('PDF storage failed');
    });

    it('reports the real job id and last server stage on timeout', async () => {
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        status: 200,
        json: async () => ({
          job_id: 'job-timeout',
          status: 'processing',
          progress: 63,
          stage: 'per_group',
          stage_message: 'Generated topic 2/3',
          created_at: '2026-09-25T00:00:00Z',
        }),
      } as Response);

      const error = await pollJobUntilDone('job-timeout', { timeoutMs: 0 })
        .then(() => null)
        .catch((value) => value as JobRunError);
      expect(error).toBeInstanceOf(JobRunError);
      expect(error?.jobId).toBe('job-timeout');
      expect(error?.lastJob?.stage).toBe('per_group');
      expect(error?.message).toContain('job-timeout');
      expect(error?.message).toContain('still recorded as processing');
    });

    it('streams only real job stages while running hierarchical notes', async () => {
      const updates: Array<{ stage?: string; progress: number }> = [];
      vi.mocked(fetch)
        .mockResolvedValueOnce({
          ok: true,
          status: 202,
          json: async () => ({ project_id: 'p1', job_id: 'job-2', status: 'pending' }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => ({
            job_id: 'job-2',
            status: 'processing',
            progress: 20,
            stage: 'model',
            stage_message: 'Spine extracted and persisted',
            created_at: '2026-09-25T00:00:00Z',
          }),
        } as Response)
        .mockResolvedValueOnce({
          ok: true,
          status: 200,
          json: async () => ({
            job_id: 'job-2',
            status: 'completed',
            progress: 100,
            stage: 'pdf',
            stage_message: 'PDF stored and artifact committed',
            created_at: '2026-09-25T00:00:00Z',
          }),
        } as Response);

      const result = await runHierarchicalNotes({
        projectId: 'p1',
        onProgress: (job) => updates.push({ stage: job.stage, progress: job.progress }),
      });
      expect(result.status).toBe('completed');
      expect(updates).toEqual([
        { stage: 'model', progress: 20 },
        { stage: 'pdf', progress: 100 },
      ]);
    });
  });

  describe('snake_case wire contract (T406)', () => {
    const okJson = (body: unknown) =>
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => body,
      } as Response);

    const lastBody = () => {
      const init = vi.mocked(fetch).mock.calls[0][1] as RequestInit | undefined;
      return JSON.parse(String(init?.body)) as Record<string, unknown>;
    };

    it('generation calls send snake_case project_id/enable_research', async () => {
      const cases: Array<{ url: string; run: () => Promise<unknown> }> = [
        { url: '/api/generation/notes', run: () => generateNotes({ projectId: 'p', query: 'q', enableResearch: true }) },
        { url: '/api/generation/flashcards', run: () => generateFlashcards({ projectId: 'p' }) },
        { url: '/api/generation/quiz', run: () => generateQuiz({ projectId: 'p' }) },
        { url: '/api/generation/summary', run: () => generateSummary({ projectId: 'p' }) },
        { url: '/api/generation/website', run: () => generateWebsite({ projectId: 'p' }) },
        { url: '/api/generation/mcq', run: () => generateMCQ({ projectId: 'p' }) },
      ];
      for (const { url, run } of cases) {
        vi.clearAllMocks();
        okJson({ project_id: 'p', mode: 'notes', content: {}, cached: false, processing_time: 0 });
        await run();
        const [hit] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit?];
        expect(hit).toContain(url);
        const body = lastBody();
        expect(body.project_id).toBe('p');
        expect(body).not.toHaveProperty('projectId');
        expect(body).not.toHaveProperty('enableResearch');
      }
    });

    it('study start omits user_id and sends snake_case', async () => {
      okJson({ session_id: 's', sessionId: 's', status: 'in_progress', total_items: 0, totalItems: 0 });
      await startStudySession({ projectId: 'p', sessionType: 'quiz' });
      const body = lastBody();
      expect(body.project_id).toBe('p');
      expect(body.session_type).toBe('quiz');
      expect(body).not.toHaveProperty('user_id');
      expect(body).not.toHaveProperty('projectId');
    });

    it('tutor + respond + complete send snake_case session_id/item_id', async () => {
      okJson({ response: 'hi', sources: [] });
      await sendTutorMessage({ sessionId: 's', message: 'hi' });
      expect(lastBody()).toEqual({ session_id: 's', message: 'hi' });

      vi.clearAllMocks();
      okJson({ item_id: 'i', is_correct: true, feedback: 'ok' });
      await submitStudyResponse({ sessionId: 's', itemId: 'i', response: 1 });
      expect(lastBody()).toEqual({ session_id: 's', item_id: 'i', response: 1 });

      vi.clearAllMocks();
      okJson({ session_id: 's', score: 100 });
      await completeStudySession({ sessionId: 's' });
      expect(lastBody()).toEqual({ session_id: 's' });
    });

    it('practice generation posts snake_case to the single MCQ endpoint', async () => {
      okJson({ project_id: 'p', mode: 'mcq', content: {}, cached: false, processing_time: 0 });
      await generatePractice({ projectId: 'p' });
      const [hit] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit?];
      expect(hit).toContain('/api/generation/mcq');
      const body = lastBody();
      expect(body.project_id).toBe('p');
      expect(body).not.toHaveProperty('projectId');
    });

    it('study website generation sends snake_case project_id/content_type', async () => {
      okJson({ website_id: 'w1', slug: 's1', title: 'Bio', content_type: 'mixed', html_content: '', preview_url: '', created_at: '' });
      await createStudyWebsite({ projectId: 'p1', contentType: 'mixed', title: 'Bio', structuredContent: { sections: [] } });
      const [hit] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit?];
      expect(hit).toContain('/api/website/generate');
      const body = lastBody();
      expect(body.project_id).toBe('p1');
      expect(body.content_type).toBe('mixed');
      expect(body.structured_content).toEqual({ sections: [] });
      expect(body).not.toHaveProperty('projectId');
      expect(body).not.toHaveProperty('contentType');
      expect(body).not.toHaveProperty('structuredContent');
    });

    it('createStudyWebsite times out and surfaces a typed timeout error', async () => {
      vi.useFakeTimers();
      try {
        // The mock MUST honour the abort signal. `apiFetch` arms a deadline
        // timer and aborts the controller; only the rejection lets its catch
        // block observe `signal.aborted` and raise the typed error. A mock
        // that ignores `signal` leaves `request()` pending forever, so the
        // assertion could never settle.
        vi.mocked(fetch).mockImplementation(
          (_input: RequestInfo | URL, init?: RequestInit) =>
            new Promise<Response>((_resolve, reject) => {
              init?.signal?.addEventListener('abort', () =>
                reject(new DOMException('The operation was aborted.', 'AbortError')));
            });

        const pending = createStudyWebsite({
          projectId: 'p',
          contentType: 'mixed',
          title: 'T',
          structuredContent: { sections: [] },
          timeoutMs: 100,
        });
        // Attach the assertion before arming the clock so the rejection is
        // never observed as unhandled.
        const assertion = expect(pending).rejects.toThrow(/timed out after/);
        await vi.advanceTimersByTimeAsync(200);
        await assertion;

        // timeoutMs is a transport concern: it must never reach the wire.
        const [, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit?];
        expect(String(init?.body)).not.toContain('timeoutMs');
        expect(String(init?.body)).not.toContain('timeout_ms');
      } finally {
        vi.useRealTimers();
      }
    });

    it('retrieval search sends snake_case project_id/top_k', async () => {
      okJson({ query: 'q', total_results: 0, chunks: [] });
      await searchContent({ query: 'q', projectId: 'p', topK: 3 });
      const [hit] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit?];
      expect(hit).toContain('/api/retrieval/search');
      const body = lastBody();
      expect(body).toEqual({ query: 'q', project_id: 'p', top_k: 3 });
      expect(body).not.toHaveProperty('projectId');
      expect(body).not.toHaveProperty('topK');
    });

    it('document bytes hit the file endpoint with the auth header', async () => {
      const blob = new Blob(['%PDF'], { type: 'application/pdf' });
      vi.mocked(fetch).mockResolvedValueOnce({ ok: true, blob: async () => blob } as Response);
      const out = await fetchDocumentBytes('doc-1');
      expect(out).toBe(blob);
      const [hit, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit?];
      expect(hit).toContain('/api/documents/doc-1/file');
      // No token in jsdom-less env: no Authorization header, and the
      // document id is never interpolated unsafely.
      expect(JSON.stringify(init?.headers ?? {})).not.toContain('undefined');
    });
  });

  describe('provider settings wire contract (user API config)', () => {
    const okJson = (body: unknown) =>
      vi.mocked(fetch).mockResolvedValueOnce({
        ok: true,
        json: async () => body,
      } as Response);

    const lastCall = () => {
      const [hit, init] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit?];
      return { hit, body: JSON.parse(String(init?.body)) as Record<string, unknown> };
    };

    it('save sends snake_case api_key/model to PUT /api/settings/providers', async () => {
      okJson({ saved: true, provider: 'groq', key_hint_masked: '....a91f', model: 'qwen/qwen3-32b' });
      await saveProviderSettings({ provider: 'groq', apiKey: 'gsk-x', model: 'qwen/qwen3-32b' });
      const { hit, body } = lastCall();
      expect(hit).toContain('/api/settings/providers');
      expect(body).toEqual({ provider: 'groq', api_key: 'gsk-x', model: 'qwen/qwen3-32b' });
      expect(body).not.toHaveProperty('apiKey');
    });

    it('test sends snake_case use_saved/api_key and never puts the key in the URL', async () => {
      okJson({ ok: true, provider: 'groq', model: 'qwen/qwen3-32b', latency_ms: 12 });
      await testProviderConnection({ provider: 'groq', apiKey: 'gsk-x', model: 'qwen/qwen3-32b' });
      const { hit, body } = lastCall();
      expect(hit).toContain('/api/settings/providers/test');
      expect(hit).not.toContain('gsk-x');
      expect(body.use_saved).toBe(false);
      expect(body.api_key).toBe('gsk-x');
      expect(body).not.toHaveProperty('apiKey');
      expect(body).not.toHaveProperty('useSaved');
    });

    it('status/models/remove hit the right endpoints', async () => {
      okJson({ active_provider: 'groq', providers: [], effective: {} });
      await getProviderStatus();
      expect(String(vi.mocked(fetch).mock.calls[0][0])).toContain('/api/settings/providers/status');

      vi.clearAllMocks();
      okJson({ providers: ['groq'], models: { groq: [] } });
      await listProviderModels({ provider: 'groq', search: 'qwen' });
      const modelsHit = String(vi.mocked(fetch).mock.calls[0][0]);
      expect(modelsHit).toContain('/api/settings/providers/models');
      expect(modelsHit).toContain('search=qwen');

      vi.clearAllMocks();
      okJson({ removed: true, provider: 'groq' });
      await removeProviderSettings('groq');
      const [delHit, delInit] = vi.mocked(fetch).mock.calls[0] as [string, RequestInit?];
      expect(delHit).toContain('/api/settings/providers/groq');
      expect(delInit?.method).toBe('DELETE');
    });
  });
});

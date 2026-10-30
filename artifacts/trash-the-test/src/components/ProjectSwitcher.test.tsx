import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import ProjectSwitcher from '../components/ProjectSwitcher';

// Mock the API services (real client reused in prod; mocked here only
// to keep the test hermetic — no network, no backend).
vi.mock('../lib/api-services', () => ({
  getProjects: vi.fn(),
  getProjectDocuments: vi.fn(),
  deleteProject: vi.fn(),
  createProject: vi.fn(),
}));

import {
  getProjects,
  getProjectDocuments,
  deleteProject,
} from '../lib/api-services';

const mockProjects = [
  {
    id: 'proj-aaa-1111',
    name: 'Data Structures',
    description: 'DS finals',
    status: 'ready' as const,
    progress: 100,
    createdAt: '2026-09-18T00:00:00',
    updatedAt: '2026-09-19T00:00:00',
  },
  {
    id: 'proj-bbb-2222',
    name: 'Quantum Physics',
    description: 'QP notes',
    status: 'processing' as const,
    progress: 40,
    createdAt: '2026-09-17T00:00:00',
    updatedAt: '2026-09-18T00:00:00',
  },
];

function renderSwitcher(props?: { open?: boolean }) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, gcTime: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <ProjectSwitcher open={props?.open ?? true} onOpenChange={() => {}} />
    </QueryClientProvider>,
  );
}

beforeEach(() => {
  vi.clearAllMocks();
  window.localStorage.clear();
  vi.mocked(getProjects).mockResolvedValue(mockProjects);
  vi.mocked(getProjectDocuments).mockResolvedValue([]);
});

describe('ProjectSwitcher', () => {
  it('lists real projects with status + truncated id', async () => {
    renderSwitcher();
    await waitFor(() => {
      expect(screen.getByText('Data Structures')).toBeDefined();
      expect(screen.getByText('Quantum Physics')).toBeDefined();
    });
    // Truncated ids render (first 8 chars + ellipsis).
    expect(screen.getByText('proj-aaa...')).toBeDefined();
  });

  it('filters the list via the search box', async () => {
    renderSwitcher();
    await waitFor(() => {
      expect(screen.getByText('Data Structures')).toBeDefined();
    });
    fireEvent.change(screen.getByLabelText('Search projects'), {
      target: { value: 'quantum' },
    });
    await waitFor(() => {
      expect(screen.getByText('Quantum Physics')).toBeDefined();
      expect(screen.queryByText('Data Structures')).toBeNull();
    });
  });

  it('stars a project and shows it under the Starred tab', async () => {
    renderSwitcher();
    await waitFor(() => {
      expect(screen.getByText('Data Structures')).toBeDefined();
    });
    fireEvent.click(screen.getByLabelText('Star Data Structures'));
    fireEvent.click(screen.getByRole('tab', { name: 'Starred' }));
    await waitFor(() => {
      expect(screen.getByText('Data Structures')).toBeDefined();
      expect(screen.queryByText('Quantum Physics')).toBeNull();
    });
  });

  it('asks for confirmation then calls DELETE on confirm', async () => {
    vi.mocked(deleteProject).mockResolvedValue(true);
    renderSwitcher();
    await waitFor(() => {
      expect(screen.getByText('Data Structures')).toBeDefined();
    });
    fireEvent.click(screen.getByLabelText('Delete Data Structures'));
    // Inline confirm appears before any API call.
    expect(deleteProject).not.toHaveBeenCalled();
    await waitFor(() => {
      expect(screen.getByText('Confirm delete')).toBeDefined();
    });
    fireEvent.click(screen.getByText('Confirm delete'));
    await waitFor(() => {
      expect(deleteProject).toHaveBeenCalledWith('proj-aaa-1111');
    });
  });

  it('shows an empty state with a creation path when zero projects exist', async () => {
    vi.mocked(getProjects).mockResolvedValue([]);
    renderSwitcher();
    await waitFor(() => {
      expect(screen.getByText(/No projects yet/)).toBeDefined();
      expect(screen.getByText('Create project')).toBeDefined();
    });
  });

  it('surfaces document-count status per project', async () => {
    vi.mocked(getProjectDocuments).mockImplementation(async (projectId: string) => [
      {
        id: `doc-${projectId}`,
        projectId,
        fileName: 'notes.pdf',
        fileType: 'pdf',
        filePath: '/x',
        processingStatus: 'completed',
        isOcrRequired: false,
        createdAt: '2026-09-18T00:00:00',
        updatedAt: '2026-09-18T00:00:00',
      },
    ] as never);
    renderSwitcher();
    await waitFor(() => {
      expect(screen.getByText('Indexed · 1 doc')).toBeDefined();
    });
  });
});

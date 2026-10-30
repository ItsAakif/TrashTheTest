import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Dashboard from '../pages/Dashboard';

// Mock the API services
vi.mock('../lib/api-services', () => ({
  getProjects: vi.fn(),
}));

import { getProjects } from '../lib/api-services';

describe('Dashboard', () => {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
      },
    },
  });

  const renderWithProviders = (component: React.ReactElement) => {
    return render(
      <QueryClientProvider client={queryClient}>
        {component}
      </QueryClientProvider>
    );
  };

  beforeEach(() => {
    vi.clearAllMocks();
    queryClient.clear();
  });

  describe('Empty States', () => {
    it('should show "No Projects Yet" when no projects exist', async () => {
      vi.mocked(getProjects).mockResolvedValue([]);

      renderWithProviders(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('No Projects Yet')).toBeDefined();
      });
    });

    it('should show loading state while fetching', () => {
      vi.mocked(getProjects).mockReturnValue(new Promise(() => {})); // Never resolves

      renderWithProviders(<Dashboard />);

      expect(screen.getByText('Loading...')).toBeDefined();
    });
  });

  describe('Error States', () => {
    it('should show error message when API fails', async () => {
      vi.mocked(getProjects).mockRejectedValue(new Error('Network error'));

      renderWithProviders(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText(/Unable to load project/i)).toBeDefined();
      });
    });
  });

  describe('Project Display', () => {
    it('should display project name when project exists', async () => {
      const mockProject = {
        id: '1',
        name: 'Quantum Physics II',
        description: 'Finals prep material',
        progress: 68,
        status: 'processing' as const,
      };

      vi.mocked(getProjects).mockResolvedValue([mockProject]);

      renderWithProviders(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText('Quantum Physics II')).toBeDefined();
      });
    });

    it('should display correct progress percentage', async () => {
      const mockProject = {
        id: '1',
        name: 'Test Project',
        description: 'Test description',
        progress: 45,
        status: 'ready' as const,
      };

      vi.mocked(getProjects).mockResolvedValue([mockProject]);

      renderWithProviders(<Dashboard />);

      await waitFor(() => {
        expect(screen.getByText(/45% COMPILED/)).toBeDefined();
      });
    });
  });

  describe('Navigation', () => {
    it('should offer project creation when no projects exist', async () => {
      vi.mocked(getProjects).mockResolvedValue([]);

      renderWithProviders(<Dashboard />);

      await waitFor(() => {
        const createButton = screen.getByText(/Create a project to get started/);
        expect(createButton).toBeDefined();
      });
    });
  });
});

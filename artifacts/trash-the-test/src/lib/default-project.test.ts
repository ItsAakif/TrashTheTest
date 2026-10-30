import { describe, it, expect } from 'vitest';
import { defaultProjectNameFor } from '../lib/api-services';

// Pure naming helper for first-run default project creation:
// "<firstNameOrUsername>'s project", fallback "My First Project".
describe('defaultProjectNameFor', () => {
  it('derives the possessive from the first name (lowercased)', () => {
    expect(defaultProjectNameFor({ name: 'John Doe', email: 'john@example.com' })).toBe("john's project");
  });

  it('uses a single-token display name as-is (lowercased)', () => {
    expect(defaultProjectNameFor({ name: 'test', email: 'other@example.com' })).toBe("test's project");
  });

  it('falls back to the email prefix when no name is set', () => {
    expect(defaultProjectNameFor({ name: '', email: 'John@example.com' })).toBe("john's project");
  });

  it('keeps only the first alphanumeric run of a dotted email prefix', () => {
    expect(defaultProjectNameFor({ name: null, email: 'john.doe42@example.com' })).toBe("john's project");
  });

  it('falls back to "My First Project" when identity is unknown', () => {
    expect(defaultProjectNameFor(null)).toBe('My First Project');
    expect(defaultProjectNameFor(undefined)).toBe('My First Project');
    expect(defaultProjectNameFor({})).toBe('My First Project');
    expect(defaultProjectNameFor({ name: '   ', email: '' })).toBe('My First Project');
    expect(defaultProjectNameFor({ name: '---', email: '' })).toBe('My First Project');
  });
});

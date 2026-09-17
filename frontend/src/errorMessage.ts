// A caught value in a catch block is typed `unknown`, not `Error` -- this is the one place that
// narrows it into a displayable string, reused everywhere a component shows a caught error.
export function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

export function restoreStillMatchesActiveDataset(
  requestDatasetId: string,
  requestEpoch: number,
  activeDatasetId: string | null,
  activeEpoch: number,
  storedDatasetId: string | null,
): boolean {
  return requestDatasetId === activeDatasetId
    && requestDatasetId === storedDatasetId
    && requestEpoch === activeEpoch
}

export function confirmsDatasetMissing(status: number | undefined, detail: string): boolean {
  return status === 404 && /\bdataset\s+(?:was\s+)?not found\b/i.test(detail)
}

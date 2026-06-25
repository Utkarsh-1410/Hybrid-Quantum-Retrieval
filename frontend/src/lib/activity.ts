import type { ActivityRecord } from "../types";

const STORAGE_KEY = "hybrid-search-activity";
const MAX_RECORDS = 100;

export function loadActivity(): ActivityRecord[] {
  try {
    const value = localStorage.getItem(STORAGE_KEY);
    return value ? (JSON.parse(value) as ActivityRecord[]) : [];
  } catch {
    return [];
  }
}

export function saveActivity(records: ActivityRecord[]): void {
  localStorage.setItem(
    STORAGE_KEY,
    JSON.stringify(records.slice(0, MAX_RECORDS)),
  );
}

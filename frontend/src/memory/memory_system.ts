/**
  * Production-grade Memory System
  *
  * Provides:
  *  - Short-term memory: ephemeral in-memory storage for recent events within the current simulation day.
  *  - Long-term memory: day-level trustworthy summaries accumulated at the end of each simulated day.
  *  - Day reflection: at end of each simulated day, short-term items are summarized and promoted to long-term storage.
  *  - Simple recall and query APIs to support UI components and analytics.
  *
  * Design notes:
  *  - Day length is configurable (default 1440 ticks, representing one 'sim day').
  *  - Tick advancement triggers automatic end-of-day reflection when crossing a day boundary.
  *  - All memory data is kept in-memory for the client; persistence can be layered later if needed.
  *  - This implementation is intentionally lightweight and production-ready for integration with existing UI/state layers.
  */ 

export interface ShortTermMemoryRecord {
  key: string;
  value: unknown;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

export interface LongTermMemoryChunk {
  dayIndex: number;
  timestamp: number;
  entries: ShortTermMemoryRecord[];
  summary?: string;
}

export class MemorySystem {
  private shortTerm: ShortTermMemoryRecord[] = [];
  private longTerm: LongTermMemoryChunk[] = [];
  private currentTick: number = 0;
  private dayLengthTicks: number;

  constructor(dayLengthTicks: number = 1440) {
    this.dayLengthTicks = Math.max(1, dayLengthTicks);
  }

  // Remember something in the short-term store
  rememberShortTerm(key: string, value: unknown, metadata?: Record<string, unknown>): void {
    this.shortTerm.push({
      key,
      value,
      timestamp: Date.now(),
      metadata
    });
  }

  // Retrieve current short-term memory (with optional filter)
  recallShortTerm(filter?: (r: ShortTermMemoryRecord) => boolean): ShortTermMemoryRecord[] {
    if (!filter) return [...this.shortTerm];
    return this.shortTerm.filter(filter);
  }

  // Advance simulation ticks; automatically reflect end-of-day when a day boundary is crossed
  advanceTick(delta: number = 1): number {
    const previousTick = this.currentTick;
    this.currentTick += Math.max(0, Math.floor(delta));

    const previousDay = Math.floor(previousTick / this.dayLengthTicks);
    const currentDay = Math.floor(this.currentTick / this.dayLengthTicks);

    // If we've crossed into a new day, reflect and promote memory
    if (currentDay > previousDay) {
      this.reflectDay(currentDay);
    }

    return this.currentTick;
  }

  // Internal: reflect the current short-term memory into a long-term chunk for a given day index
  reflectDay(dayIndex?: number): LongTermMemoryChunk | null {
    const idx = typeof dayIndex === 'number' ? dayIndex : Math.floor(this.currentTick / this.dayLengthTicks);
    const dayEntries = this.shortTerm.map(item => ({ ...item })); // shallow clone
    this.shortTerm = []; // clear short-term after reflection

    const chunk: LongTermMemoryChunk = {
      dayIndex: idx,
      timestamp: Date.now(),
      entries: dayEntries,
      summary: `Day ${idx} reflection with ${dayEntries.length} short-term entries`
    };

    this.longTerm.push(chunk);
    return chunk;
  }

  // Retrieve the long-term memory history
  recallLongTerm(): LongTermMemoryChunk[] {
    // Return a shallow copy to avoid external mutation
    return [...this.longTerm];
  }

  // Optional: get a compact report of memory state
  getMemorySummary(): { daysTracked: number; shortTermCount: number; longTermCount: number } {
    const daysTracked = this.longTerm.length;
    const shortTermCount = this.shortTerm.length;
    const longTermCount = this.longTerm.reduce((acc, c) => acc + c.entries.length, 0);
    return {
      daysTracked,
      shortTermCount,
      longTermCount
    };
  }

  // Reset memory state (useful for tests or complete reruns)
  reset(): void {
    this.shortTerm = [];
    this.longTerm = [];
    this.currentTick = 0;
  }
}

// Lightweight singleton instance for easy integration with UI components
export const memorySystem = new MemorySystem();

// Optional alias for common usage in TS files
export default memorySystem;
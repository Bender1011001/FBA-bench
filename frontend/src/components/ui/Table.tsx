import React from 'react';

export interface Column<T> {
  key: keyof T | string;
  header: string;
  render?: (row: T) => React.ReactNode;
  className?: string;
  headerClassName?: string;
  'aria-label'?: string;
}

export interface TableProps<T> {
  columns: Column<T>[];
  data: T[];
  empty?: React.ReactNode;
  getRowKey?: (row: T, index: number) => string | number;
  className?: string;
  'aria-label'?: string;
}

/**
 * Responsive, accessible table.
 * - On small screens, displays "stacked rows" with header labels via data-attributes.
 * - Preserves semantic table for screen readers.
 */
export function Table<T extends Record<string, unknown>>({
  columns,
  data,
  empty,
  getRowKey,
  className = '',
  'aria-label': ariaLabel,
}: TableProps<T>) {
  const keyFn = getRowKey ?? ((_, i) => i);

  if (!data?.length) {
    return (
      <div role="status" className="table-empty">
        {empty ?? <div className="text-sm text-gray-600">No items to display.</div>}
      </div>
    );
  }

  return (
    <div className={`table-root ${className}`} role="region" aria-label={ariaLabel}>
      <table className="w-full table-auto border-collapse">
        <thead className="table-thead">
          <tr>
            {columns.map((col, idx) => (
              <th
                key={String(col.key) + idx}
                scope="col"
                className={`table-th ${col.headerClassName ?? ''}`}
                aria-label={col['aria-label'] ?? col.header}
              >
                {col.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="table-tbody">
          {data.map((row, rowIndex) => (
            <tr key={keyFn(row, rowIndex)} className="table-tr">
              {columns.map((col, colIndex) => {
                const headerText = col.header;
                const content =
                  typeof col.render === 'function'
                    ? col.render(row)
                    : (row[col.key as keyof T] as React.ReactNode);

                return (
                  <td
                    key={String(col.key) + colIndex}
                    className={`table-td ${col.className ?? ''}`}
                    data-header={headerText}
                  >
                    {content}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

export default Table;
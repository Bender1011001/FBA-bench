import React from 'react';
import { Link } from 'react-router-dom';
import type { Experiment, Simulation } from '../../types/api';
import Badge from '../ui/Badge';
import Table, { type Column } from '../ui/Table';

export interface ExperimentListProps {
  items: Experiment[];
  onStart: (experiment: Experiment) => Promise<Simulation | void> | void;
  liveStatuses?: Record<string, Experiment['status']>; // optional realtime overrides
  isStartingId?: string | null;
}

function statusTone(status: Experiment['status']): 'neutral' | 'info' | 'success' | 'warning' | 'danger' | 'blue' | 'yellow' | 'green' | 'red' {
  switch (status) {
    case 'running':
      return 'info';
    case 'completed':
      return 'success';
    case 'failed':
      return 'danger';
    default:
      return 'neutral';
  }
}

const ExperimentList: React.FC<ExperimentListProps> = ({ items, onStart, liveStatuses = {}, isStartingId = null }) => {
  const columns: Column<Experiment>[] = [
    {
      key: 'name',
      header: 'Name',
      render: (row: Experiment) => (
        <Link
          to={`/experiments/${encodeURIComponent(row.id)}`}
          className="text-blue-700 hover:underline"
          aria-label={`Open experiment ${row.name}`}
        >
          {row.name}
        </Link>
      ),
    },
    {
      key: 'status',
      header: 'Status',
      render: (row: Experiment) => {
        const s = liveStatuses[row.id] ?? row.status;
        return <Badge tone={statusTone(s)}>{s}</Badge>;
      },
    },
    {
      key: 'created_at',
      header: 'Created',
      render: (row: Experiment) => new Date(row.created_at).toLocaleString(),
      className: 'whitespace-nowrap',
    },
    {
      key: 'updated_at',
      header: 'Updated',
      render: (row: Experiment) => new Date(row.updated_at).toLocaleString(),
      className: 'whitespace-nowrap',
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (row: Experiment) => {
        const disabled = isStartingId === row.id || (liveStatuses[row.id] ?? row.status) === 'running';
        return (
          <div className="flex items-center gap-2">
            <Link
              to={`/experiments/${encodeURIComponent(row.id)}`}
              className="inline-flex items-center rounded-md border px-2 py-1 text-xs hover:bg-gray-50"
            >
              Details
            </Link>
            <button
              type="button"
              onClick={() => onStart(row)}
              disabled={disabled}
              className="inline-flex items-center rounded-md bg-blue-600 px-2 py-1 text-xs text-white disabled:opacity-60"
              aria-label={`Start simulation for ${row.name}`}
            >
              {isStartingId === row.id ? 'Starting…' : 'Start Run'}
            </button>
          </div>
        );
      },
    },
  ];

  return (
    <Table
      aria-label="Experiments"
      columns={columns}
      data={items}
      getRowKey={(r) => r.id}
      empty={<div className="text-sm text-gray-600">No experiments yet.</div>}
    />
  );
};

export default ExperimentList;
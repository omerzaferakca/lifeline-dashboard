import React from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Decimation,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Decimation
);

const options = {
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  normalized: true,
  interaction: { mode: 'nearest', intersect: false, axis: 'x' },
  elements: { point: { radius: 0 }, line: { borderWidth: 2 } },
  plugins: {
    legend: {
      display: false,
    },
    title: {
      display: true,
      text: 'EKG',
      color: '#1f2937',
      font: {
        size: 18,
        weight: '600'
      }
    },
    decimation: { enabled: true, algorithm: 'lttb', samples: 2000, threshold: 3000 },
  },
  scales: {
    x: {
      ticks: {
        color: '#6b7280',
      },
      grid: {
        color: 'rgba(31, 41, 55, 0.08)',
      }
    },
    y: {
      ticks: {
        color: '#6b7280',
      },
      grid: {
        color: 'rgba(31, 41, 55, 0.08)',
      },
      suggestedMin: -1,
      suggestedMax: 1
    }
  }
};

const sampleData = [0.05,0.1,0.12,0.1,0.5,1,0.4,0.1,0.08,0.1,0.11,0.09,0.05,0.03,0.06,0.09,0.1,0.13,0.11,0.08,0.06,0.09,0.1,0.15,0.5,0.9,0.3,0.05,0.07,0.1,0.12,0.1,0.08,0.06,0.09,0.12,0.14,0.11,0.09,0.07,0.1,0.13,0.15,0.4,0.8,0.2,0.06,0.08,0.11,0.13,0.11,0.09,0.07,0.1,0.12,0.14,0.16,0.13,0.1,0.08,0.11,0.13,0.15,0.17,0.14,0.12,0.1,0.13,0.15,0.18,0.5,1,0.3,0.1,0.08,0.12,0.14,0.11,0.09,0.12,0.14,0.16,0.13,0.11,0.09,0.12,0.14,0.17,0.19,0.16,0.13,0.11,0.14,0.16,0.18,0.2,0.17,0.14,0.12,0.15,0.17,0.2,0.22,0.19,0.16,0.13,0.16,0.18,0.21,0.23,0.2,0.17,0.15,0.18,0.2,0.23,0.25,0.22,0.19,0.16,0.19,0.22,0.24,0.21,0.18,0.15,0.18,0.21,0.23,0.26,0.23,0.2,0.17,0.2,0.23,0.25,0.28,0.25,0.22,0.19,0.22,0.24,0.27,0.3,0.27,0.24,0.21,0.24,0.26,0.29,0.32,0.29,0.26,0.23,0.26,0.28,0.31,0.34,0.31,0.28,0.25,0.28,0.3,0.33,0.36,0.33,0.3,0.27,0.3,0.32,0.35];

const data = {
  labels: sampleData.map((_, index) => index),
  datasets: [
    {
      label: 'EKG Sinyali',
      data: sampleData,
      borderColor: '#0e7490',
      backgroundColor: 'rgba(20, 184, 166, 0.15)',
      borderWidth: 2,
      pointRadius: 0,
      fill: false,
      tension: 0.25,
    },
  ],
};

function ChartComponent() {
  return (
    <div className="chart-container ecg-paper">
      <Line options={options} data={data} />
    </div>
  );
}

export default ChartComponent;
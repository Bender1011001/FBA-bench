import React, { useState } from 'react';

interface HelpTooltipProps {
  content: string | React.ReactNode;
  title?: string;
}

const HelpTooltip: React.FC<HelpTooltipProps> = ({ content, title }) => {
  const [isVisible, setIsVisible] = useState(false);

  return (
    <div className="relative inline-block">
      <span
        className="cursor-pointer text-blue-500 hover:text-blue-700"
        onMouseEnter={() => setIsVisible(true)}
        onMouseLeave={() => setIsVisible(false)}
        onClick={() => setIsVisible(!isVisible)}
      >
        <svg
          xmlns="http://www.w3.org/2000/svg"
          className="h-4 w-4 inline-block ml-1"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth="2"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      </span>
      {isVisible && (
        <div className="absolute z-10 w-64 p-3 bg-white border border-gray-300 rounded-md shadow-lg -mt-10 left-full ml-2">
          {title && <h4 className="font-bold mb-1">{title}</h4>}
          {typeof content === 'string' ? <p className="text-sm">{content}</p> : content}
        </div>
      )}
    </div>
  );
};

export default HelpTooltip;
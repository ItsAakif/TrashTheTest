declare module "@/components/MagicCard" {
  import { ComponentType, CSSProperties, ReactNode } from "react";

  interface MagicCardProps {
    className?: string;
    style?: CSSProperties;
    children?: ReactNode;
    glowColor?: string;
    enableStars?: boolean;
    enableBorderGlow?: boolean;
    enableTilt?: boolean;
    enableMagnetism?: boolean;
    clickEffect?: boolean;
    particleCount?: number;
    spotlightRadius?: number;
    [key: string]: unknown;
  }

  const MagicCard: ComponentType<MagicCardProps>;
  export default MagicCard;
}
export {};

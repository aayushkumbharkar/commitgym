import { forwardRef, type ButtonHTMLAttributes } from 'react'
import { cva, type VariantProps } from 'class-variance-authority'
import { clsx } from 'clsx'
import { twMerge } from 'tailwind-merge'
// shadcn-compatible variants and ref forwarding; original compact implementation.
const variants=cva('btn',{variants:{variant:{default:'btn-primary',secondary:'btn-secondary',ghost:'btn-ghost'}},defaultVariants:{variant:'secondary'}})
export const Button=forwardRef<HTMLButtonElement,ButtonHTMLAttributes<HTMLButtonElement>&VariantProps<typeof variants>>(({className,variant,...props},ref)=><button ref={ref} className={twMerge(clsx(variants({variant}),className))} {...props}/>)
Button.displayName='Button'

# News Implementation Comparison

Comparison of energy news fetching between the Energy Forecast dashboard and The Digest app.

## Overview

Both apps fetch news from [NewsAPI.org](https://newsapi.org/) but with different architectures and strategies.

| App | File | Stack |
|-----|------|-------|
| Energy Forecast | `data/news_client.py` | Python/Dash |
| The Digest | `app/api/news/route.ts` | TypeScript/Next.js |

## Energy Forecast (`news_client.py`)

### Endpoint
- Uses `/everything` endpoint exclusively

### Query Strategy
Static keyword string:
```
electricity grid OR renewable energy OR solar power OR wind power OR
natural gas OR power outage OR energy prices OR utility OR ERCOT OR
power grid OR electricity demand
```

### Features
- **Caching**: None - always fetches fresh
- **Fallback**: Returns 5 hardcoded demo articles if API fails or key missing
- **Rate Limiting**: None
- **Timeout**: 10 seconds
- **Page Size**: Configurable (default 10, max 100)
- **Date Filter**: Last 7 days

### Response Format
```python
{
    "title": str,
    "description": str,
    "url": str,
    "source": str,
    "published_at": str,
    "image_url": str | None
}
```

## The Digest (`app/api/news/route.ts`)

### Endpoint
- Uses `/everything` for energy category (no native NewsAPI category)
- Uses `/top-headlines` for other categories (technology, business, etc.)

### Query Strategy
Dynamic query built from topic + subtopics:
```typescript
{
  topic: "energy",
  subtopics: [
    "electricity grid and power utilities",
    "renewable energy solar and wind",
    "energy prices and electricity demand",
    "NextEra FPL energy companies"
  ]
}
// Results in: "energy OR electricity grid and power utilities OR renewable energy solar and wind"
```

### Features
- **Caching**: In-memory Map with 15-minute TTL
- **Fallback**: Returns HTTP error codes (no demo data)
- **Rate Limiting**: 30 requests/minute per IP (sliding window)
- **Timeout**: 30 seconds (with 8-second "slow" indicator on client)
- **Page Size**: Fixed at 10
- **Date Filter**: None specified

### Response Format
```typescript
{
    id: string,
    title: string,
    summary: string,
    sourceUrl: string,
    sourceName: string,
    publishedDate: string | null,
    imageUrl: string | null,
    category: CategoryId,
    readTime: number  // estimated minutes
}
```

## Key Differences

| Aspect | Energy Forecast | The Digest |
|--------|-----------------|------------|
| Robustness | Demo fallback data | HTTP errors only |
| Caching | None (always fresh) | 15-min TTL |
| Rate Limiting | None | 30 req/min per IP |
| Query | Broad static keywords | Topic + subtopics |
| Extra Fields | None | `readTime`, `category`, `id` |
| Error Handling | Silent (returns demo) | Explicit errors |

## Recommendations

### For Energy Forecast
1. Add caching to reduce API calls (NewsAPI has rate limits)
2. Consider rate limiting if exposed to untrusted clients

### For The Digest
1. Add demo/fallback data for better UX when API fails
2. Consider adding date filtering for fresher results

## API Reference

- NewsAPI Documentation: https://newsapi.org/docs
- `/everything` endpoint: Searches all articles (requires query)
- `/top-headlines` endpoint: Top headlines by category/country

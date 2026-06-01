export function withEmbedParam(href: string, shouldEmbed: boolean) {
  if (!shouldEmbed) return href;

  const hashIndex = href.indexOf("#");
  const pathAndQuery = hashIndex >= 0 ? href.slice(0, hashIndex) : href;
  const hash = hashIndex >= 0 ? href.slice(hashIndex) : "";
  const queryIndex = pathAndQuery.indexOf("?");
  const pathname =
    queryIndex >= 0 ? pathAndQuery.slice(0, queryIndex) : pathAndQuery;
  const query = queryIndex >= 0 ? pathAndQuery.slice(queryIndex + 1) : "";
  const params = new URLSearchParams(query);

  params.set("embed", "true");

  const queryString = params.toString();
  return `${pathname}${queryString ? `?${queryString}` : ""}${hash}`;
}

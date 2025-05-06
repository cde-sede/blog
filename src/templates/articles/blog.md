# Building a Regex-Powered Static Blog Generator

In the age of heavyweight frameworks and complex build tools, there's something refreshing about going back to basics. Today, I want to share how I built a lightweight static blog generator using nothing but regex-powered string substitution and a schema-driven approach.

## The Problem with Modern Static Site Generators

Most static site generators today come with a steep learning curve. They have their own templating languages, plugin systems, and configuration formats. While powerful, they often feel like overkill for simple blogs.

What if we could build something that:
- Uses plain JavaScript for configuration
- Relies on simple string substitution rather than complex templating
- Can be understood in its entirety in less than an hour
- Still provides enough flexibility for a modern blog

## Enter Schema-Driven Site Generation

The core idea is simple: define your entire site structure in a schema file, then use regex-powered string substitution to generate the final HTML and css files.

Here's how it works:

1. A `schema.json` file defines all pages and components
2. Template files contain placeholders like `{{title}}` or `{{content}}`
3. A build script reads the schema and templates, substituting values using regex
4. The final HTML is prettified and written to the output directory

## The Build System

The heart of this approach is a straightforward build system written in JavaScript:

```javascript
// Here's the core build function that powers everything
const build = (s, data) => {
	for (const key in data) {
		const entry = data[key];
		const placeholder = `{{${key}(\\[[^]*\\])?}}`;
		const resolved = getData(entry, data, key);
		s = s.replace(new RegExp(placeholder, 'gm'), (m, c) =>
            c ? eval(c.slice(1, -1)) : data.code ? eval(data.code) : resolved);
	}
	if (data.output && data.output !== "null")
		save(join("dist", data.output !== "null" ? data.output || data.filename : "null"), s, 'utf-8');
	return s;
};
```

This function takes a template string and a data object, then replaces each placeholder with the corresponding value from the data. The magic happens in the regex pattern:

```javascript
const placeholder = `{{${key}(\\[[^]*\\])?}}`;
```

This pattern matches placeholders like `{{title}}` or more complex ones like `{{posts[resolved.map(e => e.text).join(Text.NL)]}}`, allowing for powerful transformations right in the template. In the latter example, we're mapping over an array of post objects, extracting the "text" property from each, and joining them with newlines.

## The Schema Structure

The schema is where all the configuration happens. Let me explain how it works:

1. The build system first looks for the `pages` key in the schema
2. It iterates through each string in this array and builds the corresponding object
3. A build object can have either a `filename` key (to read content from a file) or a `template` key (to use inline content)

Here's a simplified example:

```json
{
  "pages": ["home", "blog", "about"],
  "home": {
    "filename": "home.html",
    "output": "index.html",
    "title": "My Blog",
    "content": "Welcome to my blog!",
    "posts": {
      "type": "link",
      "value": "recent_posts"
    }
  },
  "recent_posts": {
    "template": "{{posts[resolved.map(e => e.text).join(Text.NL)]}}",
    "posts": [
      {
        "text": "Post number 1",
      },
      {
        "text": "Post number 2",
      }
    ]
  }
}
```

The magic happens in the `getData` method, which processes each value in the schema:
- If the value is a string or array, it's returned as is
- If the value is an object, its `type` determines how it's processed:
  - `"link"`: Searches for the specified `value` key in the schema and builds that object
  - `"file"`: Reads the file specified by the `value` key and builds it
  - `"olink"`: Searches for the `value` key in the schema and returns the object itself alongside the data
  - `"object"`: Builds the data directly
  - If no case fits, it simply returns the data

This declarative approach means we can easily:
- Reference components across the site
- Load content from external files
- Apply transformations to content
- Create complex relationships between content

## Beyond Simple Substitution

What makes this approach powerful is the ability to go beyond simple key-value substitution:

1. **Nested Components**: The schema-driven approach allows for component composition through `link` and `olink` types
2. **File Loading**: Content can be loaded from external files using the `file` type
3. **Code Evaluation**: JavaScript expressions can be evaluated within templates using the `{{key[javascriptExpression]}}` syntax
4. **Syntax Highlighting**: Built-in support for code formatting via Prism.js with automatic language detection
5. **Markdown Support**: Convert markdown to HTML with Showdown
6. **LaTeX Support**: Render mathematical formulas with KaTeX

## Performance Benefits

Since the build system is just string manipulation without any heavyweight parsing or DOM manipulation, it's blazingly fast. A complete blog with dozens of pages builds in milliseconds.

## Extensibility

The system is highly extensible because of its modular design. The `getData` function is the key to understanding how different types of content are processed:

```javascript
const getData = (data, env, name) => {
  if (typeof data === "object") {
    if (data.type === 'object')
      return fromObject({...env, ...data, [name]: undefined});
    if (data.type === 'link')
      return fromObject({...data, ...schema[getData(data.value)]});
    if (data.type === 'file')
      return fromFile(getData(data.value), data);
    if (data.type === 'olink') {
      return {...data, ...schema[data.value]};
    }
    return data
  } else return data;
};
```

Adding a new content type is as simple as adding another case to this function. For example, if you wanted to add a "fetch" type that pulls content from an API, you could easily extend the function:

```javascript
if (data.type === 'fetch')
  return fetchFromApi(getData(data.url), data);
```
## Conclusion

In a world where web development often feels needlessly complex, there's something intellectually satisfying about building a site generator that relies on nothing but string manipulation and a well-designed schema. 

However, as we've seen in the practical considerations section, this approach comes with significant trade-offs. It's more of an educational exercise than a practical solution for most use cases.

What I've learned from this experiment:
- The fundamentals of static site generation are simpler than they appear
- There's often good reason for the complexity in established tools
- Sometimes reinventing the wheel teaches you why wheels are round

If you're looking to build your own blog, you'll likely be better served by established tools. But if you're curious about how these tools work under the hood or want to challenge yourself, building a minimalist system like this one can be a rewarding experience.

Sometimes the journey of building something from scratch is more valuable than the destination. And occasionally, you might just realize that copy-pasting a header and footer wasn't such a bad solution after all.# Building a Regex-Powered Static Blog Generator

In the age of heavyweight frameworks and complex build tools, there's something refreshing about going back to basics. Today, I want to share how I built a lightweight static blog generator using nothing but regex-powered string substitution and a schema-driven approach.

## Practical Considerations: The Trade-offs

Let's be honest about the practicality of this approach. While it's an interesting exercise in minimalism, it comes with some significant trade-offs:

### Limitations

1. **Limited Scalability**: As your blog grows, the schema can become unwieldy. Managing complex relationships between dozens or hundreds of pages in a single JSON file quickly becomes difficult.

2. **Debugging Challenges**: When something goes wrong, there's no helpful error reporting system. A misplaced bracket in a JavaScript expression or an incorrect path can lead to cryptic errors.

3. **Overcomplicated for Simple Needs**: For a basic blog with a header, footer, and content area, this system is more complicated than necessary. I could have simply copy-pasted these elements across pages or used a simpler include system.

4. **Learning Curve**: Despite being relatively small, the code requires understanding several concepts: regex pattern matching, schema traversal, and JavaScript evaluation in templates.

### When It Makes Sense

This approach is best suited for:
- Small to medium-sized projects where you want complete control
- Situations where you understand the entire codebase and can quickly debug issues
- Projects where you value minimalism and independence from third-party tools
- Learning exercises to understand how static site generators work under the hood

### Alternatives

For many practical scenarios, you might be better served by:
- Using a simple copy-paste approach for headers and footers (sometimes the simplest solution is best)
- Adopting an established static site generator like Eleventy or Hugo
- Using a more structured templating system like Handlebars or Nunjucks

The reality is that this custom build system is primarily an educational exercise and a personal challenge. While it works for my needs, I wouldn't necessarily recommend it as a production-ready solution for others unless they value the learning experience over convenience.

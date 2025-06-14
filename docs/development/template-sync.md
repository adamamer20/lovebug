# Template Synchronization

This project was generated from the [pythonic-template](https://github.com/adamamer20/pythonic-template) cookiecutter template. This guide explains how to keep your project synchronized with template updates.

## Using Cruft

[Cruft](https://cruft.github.io/cruft/) allows you to update your project when the template evolves. It's included in the development dependencies.

### Installation

Cruft is already included in the development dependencies:

```bash
uv sync --all-extras
```

### Checking for Updates

To check if your project is up to date with the template:

```bash
uv run cruft check
```

### Updating from Template

To update your project with the latest template changes:

```bash
uv run cruft update
```

This will:

1. Fetch the latest template version
2. Apply changes to your project
3. Show you any conflicts that need manual resolution

### Handling Conflicts

If there are conflicts during update:

1. Review the changes carefully
2. Resolve conflicts manually
3. Test your project thoroughly
4. Commit the updates

### Configuration

The `.cruft.json` file tracks your template configuration:

- **template**: The template repository URL
- **commit**: The template commit hash your project is based on
- **context**: The cookiecutter variables used when generating your project

## Alternative: Manual Updates

If you prefer not to use Cruft, you can manually sync changes:

1. Check the [template changelog](https://github.com/adamamer20/pythonic-template/blob/main/CHANGELOG.md)
2. Review differences between template versions
3. Apply relevant changes to your project manually
4. Test thoroughly

## Best Practices

- **Regular Updates**: Check for template updates regularly
- **Test After Updates**: Always run your full test suite after updating
- **Review Changes**: Don't blindly accept all changes - review what's being updated
- **Backup First**: Consider creating a branch before major template updates
- **Custom Modifications**: Document any custom changes that might conflict with template updates

## Getting Help

If you encounter issues with template synchronization:

1. Check the [template documentation](https://github.com/adamamer20/pythonic-template)
2. Review the [Cruft documentation](https://cruft.github.io/cruft/)
3. Open an issue in the template repository
